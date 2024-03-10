import app.get_prediction as get_prediction
import os
import pytest
import boto3
from requests.models import Response
import requests_mock
import json
from io import BytesIO
from botocore.response import StreamingBody

os.environ["FEATURE_GROUP"] = "cdr_feature_group"
os.environ["MODEL_ENDPOINT"] = "cdr_model_endpoint"
os.environ["API_GATEWAY_URL"] = "http://cdr_api_gateway"

def mock_client(service_name):
    class Client():
        def get_record(self, FeatureGroupName, RecordIdentifierValueAsString):
            if FeatureGroupName == "cdr_feature_group":
                if RecordIdentifierValueAsString == "124":
                    response = {
                                "Record": [
                                    {
                                        "FeatureName": "nb_previous_loans",
                                        "ValueAsString": "5"
                                    },
                                    {
                                        "FeatureName": "avg_amount_loans_previous",
                                        "ValueAsString": "6"
                                    },
                                    {
                                        "FeatureName": "age",
                                        "ValueAsString": "25"
                                    },
                                    {
                                        "FeatureName": "years_on_the_job",
                                        "ValueAsString": "4"
                                    },
                                    {
                                        "FeatureName": "flag_own_car",
                                        "ValueAsString": "Yes"
                                    }
                                ]
                                }
                    return response


        def invoke_endpoint(self, EndpointName, Body, ContentType):
            if EndpointName == "cdr_model_endpoint" :
                if ContentType == "application/json":
                    features = {
                            "nb_previous_loans": "5",
                            "avg_amount_loans_previous" : "6",
                            "age" : "25",
                            "years_on_the_job" : "4",
                            "flag_own_car" : "Yes"
                            }
                    if Body == features :
                        body_json = [1]
                        body_encoded = json.dumps(body_json).encode("utf-8")
                        body = StreamingBody(
                                    BytesIO(body_encoded),
                                    len(body_encoded)
                                )
                        return {
                            'Body': body,
                            'ContentType': 'application/json'
                        }
    return Client()

@pytest.fixture
def mock_client_set(monkeypatch):
    monkeypatch.setattr(boto3, 'client', mock_client)

@pytest.fixture
def mock_get_response(requests_mock):
    response = {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
            },
            "body":{
                "nb_previous_loans": "5",
                "avg_amount_loans_previous" : "6",
                "age" : "25",
                "years_on_the_job" : "4",
                "flag_own_car" : "Yes"
            }
            }
    api_url = os.path.join(os.environ['API_GATEWAY_URL'], 'features')
    requests_mock.get(os.path.join(api_url, '124').replace("\\","/"), json=response)

  
def test_process_event(mock_client_set):
    event = {"headers": {
                "Content-Type": "application/json"
                },
            "payload": {
                "user_id": "124"
                }
            }

    instance = get_prediction.SagemakerManager()
    instance.process_event(event)

    assert(instance.content_type == event["headers"]["Content-Type"])
    assert(instance.user_id == event["payload"]["user_id"])

def test_get_features(mock_client_set, mock_get_response):
    event = {"headers": {
                "Content-Type": "application/json"
                },
            "payload": {
                "user_id": "124"
                }
            }
    instance = get_prediction.SagemakerManager()
    instance.process_event(event)
    instance.get_features()
    response = instance.features

    assert(response["nb_previous_loans"] == "5")
    assert(response["avg_amount_loans_previous"] == "6")
    assert(response["age"] == "25")
    assert(response["years_on_the_job"] == "4")
    assert(response["flag_own_car"] == "Yes")

def test_get_prediction(mock_client_set, mock_get_response):
    event = {"headers": {
                "Content-Type": "application/json"
                },
            "payload": {
                "user_id": "124"
                }
            }
    instance = get_prediction.SagemakerManager()
    instance.process_event(event)
    instance.get_features()
    response = instance.get_prediction()

    assert(response["statusCode"] == 200)
    assert(response["headers"]["Content-Type"] == event["headers"]["Content-Type"])
    assert(response["body"] == '[1]')

def test_lambda_handler(mock_client_set, mock_get_response):
    event = {"headers": {
                "Content-Type": "application/json"
                },
            "payload": {
                "user_id": "124"
                }
            }

    context = {
        "function_name": "get_prediction"
    }

    response = get_prediction.lambda_handler(event, context)

    assert(response["statusCode"] == 200)
    assert(response["headers"]["Content-Type"] == event["headers"]["Content-Type"])
    assert(response["body"] == '[1]')

def test_lambda_handler_invalid_request(mock_client_set,  mock_get_response):
    event = {"headers": {
                "Content-Type": "application/json"
                }
            }

    
    context = {
        "function_name": "get_prediction"
    }

    response = get_prediction.lambda_handler(event, context)

    assert(response["statusCode"] == 400)
    assert(response["message"] == "Bad request. Invalid syntax for this request was provided.")

def test_get_feature_invalid_response(mock_client_set, mock_get_response):
    event = {"headers": {
                "Content-Type": "application/json"
                },
            "payload": {
                "user_id": "123"
                }
            }

    context = {
        "function_name": "get_prediction"
    }

    response = get_prediction.lambda_handler(event, context)

    assert(response["statusCode"] == 500)
    assert(response["message"] == "Unexpected API response.")

def test_get_prediction_invalid_response(mock_client_set,  mock_get_response):
    event = {"headers": {
                "Content-Type": "application/json"
                },
            "payload": {
                "user_id": "124"
                }
            }    

    context = {
        "function_name": "get_prediction"
    }

    os.environ["MODEL_ENDPOINT"] = "wrong_endpoint"

    response = get_prediction.lambda_handler(event, context)

    assert(response["statusCode"] == 500)
    assert(response["message"] == "Unexpected Sagemaker error.")