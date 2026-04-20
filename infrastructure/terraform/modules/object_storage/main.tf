terraform {
  required_version = ">= 1.6.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Parquet datasets (data_platform), model artifacts (alpha_research),
# and backtest run outputs (backtest_engine). Buckets are versioned +
# encryption at rest is required (envs wire this through).
