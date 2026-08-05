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

variable "instances" {
  description = <<-EOT
    label -> EC2 instance type, one entry per concurrently-desired instance
    (e.g. {"ncnn-sve" = "c7g.large", "llama.cpp-sve" = "c7g.large"}). Driven
    by eval/provision.py's --label — see its module docstring.

    IMPORTANT: every apply against this config MUST be scoped with
    -target=aws_instance.labeled["<label>"] (and the matching null_resource.deploy
    target) — eval/provision.py already always does this. 
  EOT
  type        = map(string)
  default     = {}
}

variable "on_demand" {
  description = "If true, provision on-demand instead of spot — AWS won't reclaim the instance mid-run, at a higher hourly price (spot is the default: cheaper, but can be interrupted/terminated by AWS at any time with no fixed schedule)."
  type        = bool
  default     = false
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
# Key pair — shared across every labeled instance
# ---------------------------------------------------------------------------

resource "aws_key_pair" "kernel_testing" {
  key_name   = "kernel-testing-key-${formatdate("YYYY-MM-DD-hhmm", timestamp())}"
  public_key = file("~/.ssh/id_rsa.pub")
}

# ---------------------------------------------------------------------------
# Instances — one per label in var.instances. Replaces the old fixed
# aws_instance.kernel_testing ("c7g") / aws_instance.c8g pair: any number of
# instances, of any instance type, one per concurrent benchmarking job.
# ---------------------------------------------------------------------------

resource "aws_instance" "labeled" {
  for_each = var.instances

  dynamic "instance_market_options" {
    for_each = var.on_demand ? [] : [1]
    content {
      market_type = "spot"
    }
  }

  ami                    = "ami-012798e88aebdba5c" # Ubuntu 22.04 LTS arm64 us-west-2
  instance_type          = each.value
  key_name               = aws_key_pair.kernel_testing.key_name
  vpc_security_group_ids = [aws_security_group.kernel_testing.id]

  # Installs clang-18 + llvm-objdump and creates ~/arm-bench
  user_data = base64encode(file("${path.module}/setup.sh"))

  root_block_device {
    volume_size = 50
    volume_type = "gp3"
  }

  tags = {
    Name = "kernel-testing-${each.key}"
  }
}

# ---------------------------------------------------------------------------
# Deploy: wait for each instance's own bootstrap to finish.
# Source sync (allow-listed to RSYNC_ALLOWLIST — bench/, bench-trace/,
# mcp_app/, requirements.txt) and any initial build happen afterward, from
# eval/provision.py's own rsync_to()/run() calls once this resource
# completes — not here, so there's a single place that decides what gets
# synced instead of this resource's own separate deny-list rsync drifting
# out of sync with RSYNC_ALLOWLIST.
# ---------------------------------------------------------------------------

resource "null_resource" "deploy" {
  for_each = var.instances

  triggers = {
    # Re-run whenever the instance is replaced
    instance_id = aws_instance.labeled[each.key].id
  }

  connection {
    type        = "ssh"
    user        = "ubuntu"
    private_key = file("~/.ssh/id_rsa")
    host        = aws_instance.labeled[each.key].public_ip
    timeout     = "15m"
  }

  # Block until user_data (setup.sh) is done
  provisioner "remote-exec" {
    inline = ["cloud-init status --wait"]
  }
}

# ---------------------------------------------------------------------------
# Outputs — maps keyed by label, so eval/provision.py can read out.instance_public_ips[label].
# ---------------------------------------------------------------------------

output "instance_public_ips" {
  value = { for label, inst in aws_instance.labeled : label => inst.public_ip }
}

output "instance_ids" {
  value = { for label, inst in aws_instance.labeled : label => inst.id }
}

output "ssh_key_path" {
  value = "~/.ssh/id_rsa"
}
