
alias aws_cp='aws --profile cv_platform --endpoint-url=http://jssz-boss.bilibili.co s3 cp'
alias aws_ls='aws --profile cv_platform --endpoint-url=http://jssz-boss.bilibili.co s3 ls'
aws_cp  --recursive s3://cv_platform/JeremyFeng/datasets/SwinTextSpotter/ ./datasets
