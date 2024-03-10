from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import os
import json
from airflow.utils.dates import days_ago
import boto3
from sagemaker.session import Session
import pandas as pd
from sagemaker.feature_store.feature_group import FeatureGroup
import time
from airflow.models import Variable
import tarfile

REGION = Variable.get("REGION")

###### STAGES ######

def _get_config(**context) -> None:
    print("Start get_config stage.")

    with open('config/config.json') as file:
        config = json.load(file)

    context['ti'].xcom_push(key='config', value=config)

    version_tag = time.strftime('%Y%m%d-%H%M%S')    
    context['ti'].xcom_push(key='version_tag', value=version_tag)

    print("Finish get_config stage.")


def _feature_engineering(**context) -> None:
    print("Start feature_engineering stage.")

    config = context['ti'].xcom_pull(key='config')

    model_config = config['model']
    stage_config = _get_stage_config("feature_engineering", config)

    processing_config = build_processing_config_feature_engineering(model_config, stage_config)

    sm_client = _create_client('sagemaker')

    sm_client.create_processing_job(
        ProcessingInputs = processing_config['processing_inputs'],
        ProcessingOutputConfig = processing_config['processing_output_config'],
        ProcessingJobName = processing_config['process_name'],
        ProcessingResources = processing_config['processing_resources'],
        StoppingCondition = processing_config['stopping_condition'],
        AppSpecification = processing_config['app_specification'],
        NetworkConfig = processing_config['network_config'],
        RoleArn = processing_config['role']
    )

    waiter = sm_client.get_waiter('processing_job_completed_or_stopped')
    waiter.wait(ProcessingJobName=processing_config['process_name'],
            WaiterConfig={
                'Delay': 15,
                'MaxAttempts': 75
            }
        )

    final_status = sm_client.describe_processing_job(ProcessingJobName=processing_config['process_name'])
    print(final_status)

    print("Finish feature_engineering stage.")


def _register_features(**context) -> None:
    print("Start register_features stage.")

    config = context['ti'].xcom_pull(key='config')

    model_config = config['model']
    stage_config = _get_stage_config("register_features", config)

    fs_session = _feature_store_session()

    data = _download_data_for_register_features(model_config, stage_config)  

    feature_group_name = stage_config['feature_group']

    record_identifier_name = stage_config['record_identifier']
    event_time_feature_name = stage_config['loan_date']
    

    data = data.rename(
        columns={event_time_feature_name: "EventTime"}
    )

    feature_group = FeatureGroup(name = feature_group_name, sagemaker_session = fs_session)

    feature_group.load_feature_definitions(data_frame = data)

    feature_group.create(
    s3_uri = stage_config['s3_path'], 
    record_identifier_name = record_identifier_name,
    event_time_feature_name = event_time_feature_name,
    role_arn = model_config['role'],
    enable_online_store = True
    )

    _wait_for_feature_group_creation_complete(feature_group=feature_group)

    feature_group.ingest(data_frame = data, max_workers = 3, wait = True)

    print("Finish register_features stage.")


def _training(**context) -> None:
    print("Start training stage.")

    config = context['ti'].xcom_pull(key='config')

    model_config = config['model']
    stage_config = _get_stage_config("training", config)

    version_tag = context['ti'].xcom_pull(key='version_tag')
    train_process_name = model_config['model_id'] + '-training-' + version_tag
    context['ti'].xcom_push(key='train_process_name', value=train_process_name)

    processing_config = build_processing_config_training(model_config, stage_config, train_process_name)

    sm_client = _create_client('sagemaker')

    sm_client.create_training_job(
        TrainingJobName = train_process_name,
        AlgorithmSpecification= processing_config['algorithm_specification'],
        RoleArn = processing_config['role'],
        OutputDataConfig= processing_config['output_data_config'],
        ResourceConfig= processing_config['resource_config'],
        StoppingCondition= processing_config['stopping_condition'],        
        InputDataConfig = processing_config['input_data_config'],
        Environment= processing_config['environment'],
        HyperParameters= processing_config['hyperparameters']
    )

    waiter = sm_client.get_waiter('processing_job_completed_or_stopped')
    waiter.wait(ProcessingJobName = train_process_name,
            WaiterConfig={
                'Delay': 15,
                'MaxAttempts': 75
            }
        )

    final_status = sm_client.describe_processing_job(ProcessingJobName=train_process_name)
    print(final_status)

    print("Finish training stage.")


def _create_model_variant(**context) -> None:
    print("Start create_model_variant stage.")

    config = context['ti'].xcom_pull(key='config')
    version_tag = context['ti'].xcom_pull(key='version_tag')
    train_process_name = context['ti'].xcom_pull(key='train_process_name')

    model_config = config['model']
    stage_config = _get_stage_config("create_model_variant", config)    

    model_name = model_config['model_id'] + '-' + str(model_config['version']) + "-" + version_tag
    context['ti'].xcom_push(key='model_name', value=model_name)
    
    model_tar_location = os.path.join(stage_config['model']['tar_s3_path'], str(model_config['version']), train_process_name, stage_config['model']['tar_name'])
    
    source_tar_location = os.path.join(stage_config['code']['tar_s3_path'], str(model_config['version']), stage_config['code']['tar_name'])

    _build_tar_file(model_config, stage_config)

    sm_client = _create_client('sagemaker')

    sm_client.create_model(
        ModelName = model_name,
        PrimaryContainer = {
            'ContainerHostname': model_config['model_id'],
            'Image': stage_config['image'],
            'ImageConfig': {
                'RepositoryAccessMode': 'Platform'
            },
            'Mode': 'SingleModel',
            'ModelDataUrl': model_tar_location,
            'Environment': {
                'SAGEMAKER_CONTAINER_LOG_LEVEL': '20',
                'SAGEMAKER_PROGRAM': stage_config['source_code'],
                'SAGEMAKER_REGION': REGION,
                'SAGEMAKER_SUBMIT_DIRECTORY': source_tar_location
            },
        },
        ExecutionRoleArn=model_config['role'],
        EnableNetworkIsolation=False
    )

    print("Finish create_model_variant stage.")


def _create_endpoint(**context) -> None:
    print("Start create_endpoint stage.")

    config = context['ti'].xcom_pull(key='config')
    version_tag = context['ti'].xcom_pull(key='version_tag')
    model_name = context['ti'].xcom_pull(key='model_name')

    model_config = config['model']
    stage_config = _get_stage_config("create_endpoint", config)

    endpoint_name = 'ep-' + model_config['model_id'] + '-' + str(model_config['version']) + '-' + version_tag

    sm_client = _create_client('sagemaker')

    sm_client.create_endpoint_config(
        EndpointConfigName = endpoint_name,
        ProductionVariants = [
            {
                'VariantName': endpoint_name,
                'ModelName': model_name,
                'InitialInstanceCount': stage_config['instance']['count'],
                'InstanceType': stage_config['instance']['type']
            }
        ]
    )

    sm_client.create_endpoint(
        EndpointName = endpoint_name,
        EndpointConfigName = endpoint_name
    )

    waiter = sm_client.get_waiter('endpoint_in_service')
    waiter.wait(
        EndpointName=endpoint_name,
        WaiterConfig={
            'Delay': 20,
            'MaxAttempts': 31
        }
    )

    print("Finish create_endpoint stage.")


###### UTILS ######

def _get_stage_config(stage: str, config: dict):
    for resource in config['resources']:
        if resource['stage'] == stage:
            return resource

def _create_client(resource: str, region=REGION) :
    return boto3.client(resource, region_name=region)

def build_processing_config_feature_engineering(model_config: dict, stage_config: dict) -> dict:
    config = {}

    processing_inputs = []
    # Add Input Dataset Configuration
    input_conf = {
        'InputName': stage_config['input']['id'],
        'AppManaged': False,
        'S3Input': {
            'S3Uri': os.path.join(stage_config['input']['s3_path'], str(model_config['version']), stage_config['input']['file_name']),
            'LocalPath': stage_config['input']['local_path'],
            'S3DataType': 'S3Prefix',
            'S3InputMode': 'File',
            'S3DataDistributionType': 'FullyReplicated',
            'S3CompressionType': 'None'
        }
    }
    processing_inputs.append(input_conf)


    # Add Input Code Configuration
    code_conf = {
            'InputName': 'code',
            'AppManaged': False,
            'S3Input': {
                'S3Uri': os.path.join(model_config['location'], stage_config['code']['source_path'], stage_config['code']['file_name']),
                'LocalPath': stage_config['code']['local_path'],
                'S3DataType': 'S3Prefix',
                'S3InputMode': 'File',
                'S3DataDistributionType': 'FullyReplicated',
                'S3CompressionType': 'None'
            }
        }
    processing_inputs.append(code_conf)
    

    processing_outputs = []
    # Add Output Dataset Configuration
    output_conf =  {
        'OutputName': stage_config['output']['id'],
        'S3Output': {
            'S3Uri': os.path.join(stage_config['output']['s3_path'], str(model_config['version'])),
            'LocalPath': stage_config['output']['local_path'],
            'S3UploadMode': 'EndOfJob'
        },
        'AppManaged': False
    }
    processing_outputs.append(output_conf)

        
    config['processing_output_config'] = {
        'Outputs': processing_outputs,
    }

    config['processing_resources'] = {
        'ClusterConfig': {
            'InstanceType': stage_config['instance']['type'], 
            'InstanceCount': stage_config['instance']['count'],
            'VolumeSizeInGB': stage_config['instance']['size_GB']
        }
    }

    config['stopping_condition'] = {
        'MaxRuntimeInSeconds': stage_config['instance']['max_runtime']
    }

    config['app_specification'] = {
        'ImageUri': stage_config['image'],
        'ContainerEntrypoint': [
            '/opt/program/submit'
        ],
        'ContainerArguments': [
            os.path.join(stage_config['code']['local_path'], stage_config['code']['file_name']),
            "--input_data",
            os.path.join(stage_config['input']['local_path'], stage_config['input']['file_name']),
            "--output_data",
            stage_config['output']['local_path']
        ]
    }

    config['network_config'] = {
        'EnableInterContainerTrafficEncryption': False,
        'EnableNetworkIsolation': False
    }

    config['role'] = model_config['role']

    config['process_name'] = model_config['model_id'] + '-feature_engineering-' + model_config['version']


    return config

def build_processing_config_training(model_config: dict, stage_config: dict, train_process_name: str) -> dict:
    config = {}

    _build_tar_file(model_config, stage_config)

    train_tar_location = os.path.join(stage_config['code']['tar_s3_path'],str(model_config['version']), stage_config['code']['tar_name'])
    
    processing_inputs = []
    # Add Input Dataset Configuration
    input_conf = {
        'ChannelName': stage_config['input']['id'],
        'DataSource': {
            'S3DataSource': {
                'S3DataType': 'S3Prefix',
                'S3Uri':  os.path.join(stage_config['input']['s3_path'],str(model_config['version'])),
                'S3DataDistributionType': 'FullyReplicated'
            }
        },
        'ContentType': stage_config['input']['content_type'],
        'CompressionType': 'None'
    }
    processing_inputs.append(input_conf)

    config['input_data_config'] = processing_inputs

    config['algorithm_specification'] = {
        'TrainingImage': stage_config['image'],
        'TrainingInputMode': 'File'
    }

    config['role'] = model_config['role']

    config['output_data_config'] = {
        'S3OutputPath': os.path.join(stage_config['output']['s3_path'], str(model_config['version']))
    }

    config['resource_config'] = {
        'InstanceType': stage_config['instance']['type'], 
        'InstanceCount': stage_config['instance']['count'],
        'VolumeSizeInGB': stage_config['instance']['size_GB']
    }

    config['stopping_condition'] = {
        'MaxRuntimeInSeconds': stage_config['instance']['max_runtime']
    }

    config['environment'] = {
        'SAGEMAKER_PROGRAM': stage_config['code']['file_name'],
        'SAGEMAKER_SUBMIT_DIRECTORY': train_tar_location
    }
    
    config['hyperparameters'] = {
        'sagemaker_program': stage_config['code']['file_name'],
        'sagemaker_submit_directory': train_tar_location,
        'sagemaker_container_log_level': '20',
        'sagemaker_job_name': train_process_name,
        'sagemaker_region': REGION
    }

    return config

def _build_tar_file(model_config: dict, stage_config: dict) -> None:

    s3_client = _create_client('s3')
    s3_resource = boto3.resource('s3')

    # Input
    input_bucket_name, input_prefix = _bucketname_and_key(model_config['location'])
    source_bucket = s3_resource.Bucket(input_bucket_name)
    source_path = os.path.join(input_prefix, stage_config['code']['source_path'])

    source_code = _download_files_in_folder(source_bucket, source_path)

    with tarfile.open(stage_config['code']['tar_name'], "w:gz") as tar:
        for item in source_code:  
            tar.add(item)

    # Output
    output_tar_location = os.path.join(stage_config['code']['tar_s3_path'], str(model_config['version']), stage_config['code']['tar_name'])
    output_bucket_name, output_prefix = _bucketname_and_key(output_tar_location)
    s3_client.upload_file(stage_config['code']['tar_name'], output_bucket_name, output_prefix)
   
def _download_files_in_folder(bucket, path):
    items = []
    for obj in bucket.objects.filter(Prefix= path):
        target = obj.key.split('/')[-1]
        if target == '': continue
        items.append(target)
        bucket.download_file(obj.key, target)
    return items

def _bucketname_and_key(s3_uri: str) -> tuple:
    splitted = s3_uri.split('/')
    return splitted[2], '/'.join(splitted[3:])

def _feature_store_session() -> Session:
    boto_session = boto3.Session(region_name=REGION)
    sagemaker_client = boto_session.client(service_name="sagemaker", region_name=REGION)
    feature_store_runtime = boto_session.client(service_name="sagemaker-featurestore-runtime", region_name=REGION)

    feature_store_session = Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_featurestore_runtime_client=feature_store_runtime
    )

    return feature_store_session

def _download_data_for_register_features(model_config: dict, stage_config: dict) -> pd.DataFrame:
    s3_resource = boto3.resource("s3", region_name=REGION)

    # Input
    input_bucket_name, input_prefix = _bucketname_and_key(stage_config['input']['s3_path'])
    input_bucket = s3_resource.Bucket(input_bucket_name)

    items = _download_files_in_folder(input_bucket, input_prefix)
    
    dataframes = []
    for item in items:
        data_partition = pd.read_parquet(item)
        dataframes.append(data_partition)
    
    data = pd.concat(dataframes, ignore_index=True)
    
    return data

def _wait_for_feature_group_creation_complete(feature_group: FeatureGroup) -> None:
    status = feature_group.describe().get("FeatureGroupStatus")
    while status == "Creating":
        print("Waiting for Feature Group Creation")
        time.sleep(5)
        status = feature_group.describe().get("FeatureGroupStatus")
    if status != "Created":
        raise RuntimeError(f"Failed to create feature group {feature_group.name}")
    print(f"FeatureGroup {feature_group.name} successfully created.")

###### DAG ######

with DAG(
    'training_dag',
    start_date=days_ago(10),
    schedule_interval=None,
    default_args={'owner': 'kueski'},
    description='Training DAG for Credit Default Risk Model',
    tags=['training', 'CreditDefaultRisk'],
) as dag:

    get_config = PythonOperator(task_id='get_config', python_callable=_get_config)
    feature_engineering = PythonOperator(task_id='feature_engineering', python_callable=_feature_engineering)
    register_features = PythonOperator(task_id='register_features', python_callable=_register_features)
    training = PythonOperator(task_id='training', python_callable=_training)
    create_model_variant = PythonOperator(task_id='create_model_variant', python_callable=_create_model_variant)
    create_endpoint = PythonOperator(task_id='create_endpoint', python_callable=_create_endpoint)

    get_config >> feature_engineering >> register_features >> training >> create_model_variant >> create_endpoint