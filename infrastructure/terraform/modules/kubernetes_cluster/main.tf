terraform {
  required_version = ">= 1.6.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Cloud-native Kubernetes cluster for mid-frequency + research + agent
# workloads. HFT engine never runs here (see
# docs/architecture/hft-colocation-phase-8.md).
