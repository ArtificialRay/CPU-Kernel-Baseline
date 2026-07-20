terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-west-2"
}

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

variable "build_target" {
  description = "arm-bench make target (scalar, neon, sve, sve2, sme2, all, ...)"
  default     = "sve"
  # c7g  = Graviton3 (Neoverse V1)  — SVE at 256-bit (no SVE2, no SME)
  # c8g  = Graviton4 (Neoverse V2)  — SVE2 at 128-bit (no SME)
  # No AWS instance type supports SME/SME2 as of early 2026.
}

variable "instance_type" {
  description = "EC2 instance type (must be arm64 / Graviton)"
  default     = "c7g.large"
}


# ---------------------------------------------------------------------------
# Security group — SSH only
# ---------------------------------------------------------------------------

resource "aws_security_group" "kernel_testing" {
  name = "kernel-testing-sg"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ---------------------------------------------------------------------------
# Key pair
# ---------------------------------------------------------------------------

resource "aws_key_pair" "kernel_testing" {
  key_name   = "kernel-testing-key-${formatdate("YYYY-MM-DD-hhmm", timestamp())}"
  public_key = file("~/.ssh/id_rsa.pub")
}

# ---------------------------------------------------------------------------
# Instance
# ---------------------------------------------------------------------------

resource "aws_instance" "kernel_testing" {
  instance_market_options {
    market_type = "spot"
  }

  ami                    = "ami-012798e88aebdba5c" # Ubuntu 22.04 LTS arm64 us-west-2
  instance_type          = var.instance_type
  key_name               = aws_key_pair.kernel_testing.key_name
  vpc_security_group_ids = [aws_security_group.kernel_testing.id]

  # Installs clang-18 + llvm-objdump and creates ~/arm-bench
  user_data = base64encode(file("${path.module}/setup.sh"))

  root_block_device {
    volume_size = 50
    volume_type = "gp3"
  }

  tags = {
    Name = "kernel-testing"
  }
}

# ---------------------------------------------------------------------------
# Deploy: wait for the instance's own bootstrap to finish.
# Source sync (allow-listed to RSYNC_ALLOWLIST — bench/, bench-trace/,
# mcp_app/, requirements.txt) and any initial build happen afterward, from
# eval/provision.py's own rsync_to()/run() calls once this resource
# completes — not here, so there's a single place that decides what gets
# synced instead of this resource's own separate deny-list rsync drifting
# out of sync with RSYNC_ALLOWLIST.
# ---------------------------------------------------------------------------

resource "null_resource" "deploy" {
  triggers = {
    # Re-run whenever the instance is replaced
    instance_id = aws_instance.kernel_testing.id
  }

  connection {
    type        = "ssh"
    user        = "ubuntu"
    private_key = file("~/.ssh/id_rsa")
    host        = aws_instance.kernel_testing.public_ip
    timeout     = "15m"
  }

  # Block until user_data (setup.sh) is done
  provisioner "remote-exec" {
    inline = ["cloud-init status --wait"]
  }
}

# ---------------------------------------------------------------------------
# c8g instance (Graviton4, SVE2 128-bit)
# ---------------------------------------------------------------------------

resource "aws_instance" "c8g" {
  instance_market_options {
    market_type = "spot"
  }

  ami                    = "ami-012798e88aebdba5c" # Ubuntu 22.04 LTS arm64 us-west-2
  instance_type          = var.instance_type
  key_name               = aws_key_pair.kernel_testing.key_name
  vpc_security_group_ids = [aws_security_group.kernel_testing.id]
  user_data              = base64encode(file("${path.module}/setup.sh"))

  root_block_device {
    volume_size = 50
    volume_type = "gp3"
  }

  tags = {
    Name = "kernel-testing-c8g"
  }
}

resource "null_resource" "deploy_c8g" {
  triggers = {
    instance_id = aws_instance.c8g.id
  }

  connection {
    type        = "ssh"
    user        = "ubuntu"
    private_key = file("~/.ssh/id_rsa")
    host        = aws_instance.c8g.public_ip
    timeout     = "15m"
  }

  provisioner "remote-exec" {
    inline = ["cloud-init status --wait"]
  }
}

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

output "instance_public_ip" {
  value = aws_instance.kernel_testing.public_ip
}

output "ssh_command" {
  value = "ssh -i ~/.ssh/id_rsa ubuntu@${aws_instance.kernel_testing.public_ip}"
}

output "run_example" {
  value = "ssh -i ~/.ssh/id_rsa ubuntu@${aws_instance.kernel_testing.public_ip} './arm-bench/build/${var.build_target}/bin/simd_loops -k 1 -n 10'"
}

output "instance_id" {
  value = aws_instance.kernel_testing.id
}

output "ssh_key_path" {
  value = "~/.ssh/id_rsa"
}

output "c8g_public_ip" {
  value = aws_instance.c8g.public_ip
}

output "c8g_instance_id" {
  value = aws_instance.c8g.id
}
