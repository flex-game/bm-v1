#!/bin/bash

# Get AWS account ID
aws_account_id=$(aws sts get-caller-identity --query Account --output text)
region="us-east-1"  # or your region

# Create ECR repository if it doesn't exist
aws ecr create-repository --repository-name bm-v1-training || true

# Login to ECR
aws ecr get-login-password --region $region | docker login --username AWS --password-stdin $aws_account_id.dkr.ecr.$region.amazonaws.com

# Build image
echo "Building Docker image..."
docker build -t bm-v1-training . || { echo "Docker build failed"; exit 1; }

# Tag image
echo "Tagging image..."
docker tag bm-v1-training:latest $aws_account_id.dkr.ecr.$region.amazonaws.com/bm-v1-training:latest

# Push to ECR
echo "Pushing to ECR..."
docker push $aws_account_id.dkr.ecr.$region.amazonaws.com/bm-v1-training:latest

# Save the image URI to a variable
image_uri="$aws_account_id.dkr.ecr.$region.amazonaws.com/bm-v1-training:latest"

echo "Image URI: $image_uri"
echo "You can add this to your .env file as:"
echo "SAGEMAKER_IMAGE_URI=$image_uri" 