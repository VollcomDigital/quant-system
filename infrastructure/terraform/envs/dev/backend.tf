terraform {
  backend "s3" {
    bucket  = "quant-system-tfstate-dev"
    key     = "envs/dev/terraform.tfstate"
    region  = "us-east-1"
    encrypt = true
  }
}
