terraform {
  backend "s3" {
    bucket         = "quant-system-tfstate-paper"
    key            = "envs/paper/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "quant-system-tfstate-lock-paper"
  }
}
