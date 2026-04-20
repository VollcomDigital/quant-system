terraform {
  required_version = ">= 1.6.0"

  required_providers {
    vault = {
      source  = "hashicorp/vault"
      version = "~> 4.0"
    }
  }
}

# Secrets pathways for:
# - broker API keys (alpaca, ibkr, polygon, tiingo, ...)
# - RPC provider keys (alchemy, infura)
# - web_control_plane session signing keys
# Real resources land alongside the deployment; the module contract is
# what Phase 9 commits.
