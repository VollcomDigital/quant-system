# Input variables for this module. Real values are wired per-env.
variable "environment" {
  description = "Target environment (dev / paper / production)."
  type        = string
}

variable "tags" {
  description = "Resource tags."
  type        = map(string)
  default     = {}
}
