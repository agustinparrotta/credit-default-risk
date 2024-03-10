import os
import boto3
import logging
import json
import boto3
import requests
from functools import wraps

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Decorators
def notify_cloudwatch(function):
    @wraps(function)
    def wrapper(event, context):
        logger.info(f'"{context["function_name"]}" - entry.\nIncoming event: "{event}" ')
        result = function(event, context)
        logger.info(f'"{context["function_name"]}" - exit.\nResult: "{result}"')
        return result
    return wrapper

class SagemakerManager:
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)

        self.sm_client = boto3.client("sagemaker-runtime")
        self.model_endpoint = os.environ['MODEL_ENDPOINT']
        self.url_features_endpoint = os.path.join(os.environ['API_GATEWAY_URL'], 'features').replace("\\","/")

    def process_event(self, event) -> None:
        self.logger.debug("Event %s", json.dumps(event))

        # Get posted body and content type
        self.content_type = event["headers"].get("Content-Type")
        if self.content_type.startswith("text/csv"):
            payload = json.loads(event["payload"])
            self.user_id = payload["user_id"]
        elif self.content_type.startswith("application/json"):
            self.user_id = event["payload"]["user_id"]
        
    def get_features(self):
        response = requests.get(os.path.join(self.url_features_endpoint, self.user_id).replace("\\","/"))
        response = json.loads(response.text)
        response = response['body']
        self.features = response
    
    def get_prediction(self) -> dict:
        # Invoke the endpoint with full multi-line payload
        response = self.sm_client.invoke_endpoint(
            EndpointName=self.model_endpoint,
            Body=self.features,
            ContentType=self.content_type,
        )
        # Return predictions as JSON dictionary instead of CSV text
        predictions = response["Body"].read().decode("utf-8")
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": self.content_type,
            },
            "body": predictions,
        }

@notify_cloudwatch
def lambda_handler(event, context):
    sagemaker_instance = SagemakerManager()

    try:
        sagemaker_instance.process_event(event)
    except Exception as e:
        message = "Bad request. Invalid syntax for this request was provided."
        logger.error(message)
        logger.error(e)
        return {"statusCode": 400, "message": message} 

    try:
        sagemaker_instance.get_features()
    except Exception as e:
        message = "Unexpected API response."
        logger.error(message)
        logger.error(e)
        return {"statusCode": 500, "message": message}  

    try:
        return sagemaker_instance.get_prediction()
    except Exception as e:
        message = "Unexpected Sagemaker error."
        logger.error(message)
        logger.error(e) 
        return {"statusCode": 500, "message": message}