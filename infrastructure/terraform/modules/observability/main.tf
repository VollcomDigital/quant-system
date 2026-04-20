terraform {
  required_version = ">= 1.6.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# OTel collector + Kafka topics for shared_lib.logging JSON output,
# shared_lib.telemetry spans, and trading_system anomaly/fills events.
