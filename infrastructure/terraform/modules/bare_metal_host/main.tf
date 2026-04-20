terraform {
  required_version = ">= 1.6.0"
  # Bare-metal provisioning uses a provider suited to the vendor
  # (Equinix Metal, custom tooling, ...). Phase 9 pins the AWS
  # provider so IAM / KMS plumbing to the HFT host works uniformly.
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Bare-metal HFT hosts (see hft-colocation-phase-8.md): CPU pinning,
# hugepages, kernel-bypass NIC, PTP sync. Hardware procurement is out
# of scope; the module exposes IAM + KMS + observability plumbing.
