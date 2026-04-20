terraform {
  required_version = ">= 1.6.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

locals {
  environment = "paper"
  tags = {
    environment = local.environment
    managed_by  = "terraform"
  }
}

module "kms_signing" {
  source      = "../../modules/kms_signing"
  environment = local.environment
  tags        = local.tags
}

module "vault_secrets" {
  source      = "../../modules/vault_secrets"
  environment = local.environment
  tags        = local.tags
}

module "object_storage" {
  source      = "../../modules/object_storage"
  environment = local.environment
  tags        = local.tags
}

module "observability" {
  source      = "../../modules/observability"
  environment = local.environment
  tags        = local.tags
}

module "kubernetes_cluster" {
  source      = "../../modules/kubernetes_cluster"
  environment = local.environment
  tags        = local.tags
}
