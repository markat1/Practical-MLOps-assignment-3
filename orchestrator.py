import boto3
import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# Define AWS credentials and region
region = os.getenv('REGION')
role_arn = os.getenv('SAGEMAKER_ROLE')

#Define ECR image and S3 paths
ecr_image = os.getenv('ECR_IMAGE')
s3_input_train = os.getenv('S3_INPUT_TRAIN')
s3_output_path = os.getenv('S3_OUTPUT_PATH')
training_job_name = os.getenv('TRAINING_JOB_NAME')


# Initialize Boto3 SageMaker client
sagemaker_client = boto3.client("sagemaker", region_name=region)

# Define training job name

# Create training job configuration
training_job_config = {
    "TrainingJobName": training_job_name,
    "AlgorithmSpecification": {"TrainingImage": ecr_image, "TrainingInputMode": "File"},
    "RoleArn": role_arn,
    "InputDataConfig": [
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": s3_input_train,
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
            "ContentType": "text/csv",
            "InputMode": "File",
        }
    ],
    "OutputDataConfig": {"S3OutputPath": s3_output_path},
    "ResourceConfig": {
        "InstanceType": "ml.m5.large",
        "InstanceCount": 1,
        "VolumeSizeInGB": 50,
    },
    "StoppingCondition": {"MaxRuntimeInSeconds": 3600},
    "HyperParameters": {"n_estimators": "100"},
}

# Create the training job
response = sagemaker_client.create_training_job(**training_job_config)

# Print the response
print(response)