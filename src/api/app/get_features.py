import os
import boto3
import logging
import json
from functools import wraps

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Decorators
def notify_cloudwatch(function):
    @wraps(function)
    def wrapper(event, context):
        logger.info(f'"{context["function_name"]}" - entry.\nIncoming event: "{event}"')
        result = function(event, context)
        logger.info(f'"{context["function_name"]}" - exit.\nResult: "{result}"')
        return result
    return wrapper

class SagemakerManager:
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)

        self.sm_client = boto3.client("sagemaker-featurestore-runtime")
        self.feature_group = os.environ["FEATURE_GROUP"]

    def process_event(self, event) -> None:
        self.logger.debug("Event %s", json.dumps(event))
        
        # Get event and content type
        self.content_type = event["headers"].get("Content-Type")
        if self.content_type.startswith("text/csv"):
            payload = json.loads(event["payload"])
            self.user_id = payload["user_id"]
        elif self.content_type.startswith("application/json"):
            self.user_id = event["payload"]["user_id"]

    def get_record(self) -> dict:
        # Get features from Sagemaker
        response = self.sm_client.get_record(
            FeatureGroupName=self.feature_group,
            RecordIdentifierValueAsString=self.user_id
        )
        # Return features as list of JSON dictionary
        features = response["Record"]

        features_transformed = {feature["FeatureName"]: feature["ValueAsString"] for feature in features}
 
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": self.content_type,
            },
            "body": features_transformed,
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
        return sagemaker_instance.get_record()
    except Exception as e:
        message = "Unexpected Sagemaker error."
        logger.error(message)
        logger.error(e) 
        return {"statusCode": 500, "message": message}
