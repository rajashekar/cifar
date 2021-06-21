# CIFAR on Google Cloud

## Creating Google Cloud instance
Google deep learning vm can be created by below command
```
export IMAGE_FAMILY="pytorch-latest-gpu"
export ZONE="us-west1-b"
export INSTANCE_NAME="dl-test-automation"
gcloud compute instances create $INSTANCE_NAME \
  --zone=$ZONE \
  --image-family=$IMAGE_FAMILY \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --accelerator="type=nvidia-tesla-t4,count=1" \
  --metadata="install-nvidia-driver=True" \
  --no-address \
  --subnet projects/shared-vpc-admin/regions/us-west1/subnetworks/prod-us-west1-01 
  ```

## To ssh to above google instance
```
gcloud compute ssh \
--project wmt-sams-fe-eng \
--internal-ip \
--zone $ZONE
```

## Opening Jupyter lab from above instance
```
gcloud compute ssh \
--project wmt-sams-fe-eng \
--internal-ip \
--zone $ZONE \
$INSTANCE_NAME -- -L 8080:localhost:8080
```
Go to localhost:8080 to get Jupyter lab 

## To run Cifar example
[model_cifar.pt](model_cifar.pt) is saved model trained from [cifar.ipynb](cifar.ipynb) <br>
[app.py](app.py) will load model_cifar.pt and will provide prediction on given images. 

## To start flask server for inference
Use base environment
```
conda activate base
```
Run below command to up flask instance
```
FLASK_ENV=development FLASK_APP=app.py flask run
```

## Examples
Request
```
curl -X POST -H 'Content-Type: multipart/form-data' http://localhost:5000/predict -F "file=@frog.jpeg"
```
Response
```
{
  "class_name": "frog"
}
```
