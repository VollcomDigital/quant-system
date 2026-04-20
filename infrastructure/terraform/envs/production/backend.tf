terraform {
  backend "s3" {
    bucket         = "quant-system-tfstate-production"
    key            = "envs/production/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    kms_key_id     = "arn:aws:kms:us-east-1:PRODUCTION:key/tfstate"
    dynamodb_table = "quant-system-tfstate-lock-production"
  }
}
