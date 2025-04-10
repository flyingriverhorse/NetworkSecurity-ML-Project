Network Security project involves for Phishing data

Update secrets in GITHUB for below in settings in secrets and variables:

aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
aws-region: ${{ secrets.AWS_REGION }}
ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
ECR_REPOSITORY: ${{ secrets.ECR_REPOSITORY_NAME }}

uvicorn http://127.0.0.1:8000/docs to access!
