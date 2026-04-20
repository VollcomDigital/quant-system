# AWS KMS signing keys used by trading_system.gateways.web3 SigningClient
# and by treasury signing flows. See docs/adr/0006-execution-signing-
# custody-and-kill-switches.md.

terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Placeholder: one signing key per signer role (trading_signer,
# treasury_signer, kill_switch_signer). Real resources land with the
# Phase 9 deployment; Phase 9 pins the module contract.
locals {
  signer_roles = [
    "trading_signer",
    "treasury_signer",
    "kill_switch_signer",
  ]
}
