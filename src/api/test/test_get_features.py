import app.get_features as get_features
import os
import pytest
import boto3

os.environ["FEATURE_GROUP"] = "cdr_feature_group"

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
    return Client()

@pytest.fixture
def mock_client_set(monkeypatch):
    monkeypatch.setattr(boto3, 'client', mock_client)

def test_process_event(mock_client_set):
    event = {"headers": {
                "Content-Type": "application/json"
                },
            "payload": {
                "user_id": "124"
                }
            }

    instance = get_features.SagemakerManager()
    instance.process_event(event)

    assert(instance.content_type == event["headers"]["Content-Type"])
    assert(instance.user_id == event["payload"]["user_id"])

def test_get_record(mock_client_set):
    event = {"headers": {
                "Content-Type": "application/json"
                },
            "payload": {
                "user_id": "124"
                }
            }

    instance = get_features.SagemakerManager()
    instance.process_event(event)
    response = instance.get_record()

    assert(response["statusCode"] == 200)
    assert(response["headers"]["Content-Type"] == event["headers"]["Content-Type"])
    assert(response["body"]["nb_previous_loans"] == "5")
    assert(response["body"]["avg_amount_loans_previous"] == "6")
    assert(response["body"]["age"] == "25")
    assert(response["body"]["years_on_the_job"] == "4")
    assert(response["body"]["flag_own_car"] == "Yes")

def test_lambda_handler(mock_client_set):
    event = {"headers": {
                "Content-Type": "application/json"
                },
            "payload": {
                "user_id": "124"
                }
            }

    context = {
        "function_name": "get_features"
    }

    response = get_features.lambda_handler(event, context)

    assert(response["statusCode"] == 200)
    assert(response["headers"]["Content-Type"] == event["headers"]["Content-Type"])
    assert(response["body"]["nb_previous_loans"] == "5")
    assert(response["body"]["avg_amount_loans_previous"] == "6")
    assert(response["body"]["age"] == "25")
    assert(response["body"]["years_on_the_job"] == "4")
    assert(response["body"]["flag_own_car"] == "Yes")

def test_lambda_handler_invalid_request(mock_client_set):
    event = {"headers": {
                "Content-Type": "application/json"
                }
            }

    
    context = {
        "function_name": "get_features"
    }

    response = get_features.lambda_handler(event, context)

    assert(response["statusCode"] == 400)
    assert(response["message"] == "Bad request. Invalid syntax for this request was provided.")

def test_lambda_handler_invalid_response(mock_client_set):
    event = {"headers": {
                "Content-Type": "application/json"
                },
            "payload": {
                "user_id": "123"
                }
            }

    context = {
        "function_name": "get_features"
    }

    response = get_features.lambda_handler(event, context)

    assert(response["statusCode"] == 500)
    assert(response["message"] == "Unexpected Sagemaker error.")