terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  # Refuse to run under credentials for any account but the one the current
  # workspace lives in — see var.workspace_account_ids.
  allowed_account_ids = length(var.workspace_account_ids) == 0 ? null : [var.workspace_account_ids[terraform.workspace]]
}

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

variable "workspace_account_ids" {
  description = <<-EOT
    Terraform workspace -> id of the AWS account that workspace lives in, e.g.
    {"default" = "111111111111", "scs-lti-l3" = "222222222222"}.

    A workspace isolates state, not credentials. Run one workspace's state
    under another account's AWS_PROFILE and terraform finds none of its
    resources, drops them from state and re-creates them in the wrong account,
    leaving the originals running with nothing tracking them.

    Once this map is non-empty the provider only accepts the account listed for
    the current workspace, and a workspace with no entry is an error rather
    than an unguarded run. Empty (the default) keeps the old behaviour: no
    check. Set it via TF_VAR_workspace_account_ids so account ids stay out of
    the repo.
  EOT
  type        = map(string)
  default     = {}
}

variable "build_target" {
  description = "arm-bench make target (scalar, neon, sve, sve2, sme2, all, ...)"
  default     = "sve"
  # c7g          = Graviton3 (Neoverse V1) — SVE at 256-bit (no SVE2, no SME)
  # c8g          = Graviton4 (Neoverse V2) — SVE2 at 128-bit (no SME)
  # mac-m4.metal = Apple M4                — SME2 at 512-bit streaming SVL
  # No Graviton generation implements SME/SME2 (it is optional in Armv9.2-A and
  # AWS did not take it), so sme2 is the one tier that lands on Apple silicon —
  # see the "Mac tier" section below and config/kernel_contracts.yaml's isa.sme2.
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
  description = "If true, provision on-demand instead of spot — AWS won't reclaim the instance mid-run, at a higher hourly price (spot is the default: cheaper, but can be interrupted/terminated by AWS at any time with no fixed schedule). Ignored for Mac labels, which have no spot market."
  type        = bool
  default     = false
}

variable "mac_host_ids" {
  description = <<-EOT
    label -> id of an ALREADY-ALLOCATED Dedicated Host to place that label's
    EC2 Mac instance on (e.g. {"ncnn-sme2" = "h-029b7735a4392cedc"}).

    Required for every mac label: this config only CONSUMES host ids, it never
    allocates or releases hosts.
  EOT
  type        = map(string)
  default     = {}
}

variable "aws_region" {
  description = "AWS region for everything in this config, set from the current workspace's `aws_region` in provisioning/workspaces.json (provision.py injects TF_VAR_aws_region). No default. The AMI ids below are region-specific — changing this also means swapping them."
  type        = string
}

variable "namespace" {
  description = <<-EOT
    Suffix for the account-global names this config creates (security group,
    key pairs), so several Terraform states — e.g. one per git worktree — can
    coexist in one AWS account without InvalidGroup.Duplicate. Empty keeps
    the original "kernel-testing-sg". Set via TF_VAR_namespace, e.g. "sweep".
  EOT
  type        = string
  default     = ""
}

variable "ssh_key_files" {
  description = <<-EOT
    label -> private key path used for that label's instance (default
    ~/.ssh/id_rsa). The matching public key is registered as that label's EC2
    key pair from the same path with ".pem" stripped and ".pub" appended
    (~/.ssh/mac-m4-key.pem -> ~/.ssh/mac-m4-key.pub). Lets Graviton labels use
    id_rsa and Mac labels use mac-m4-key.pem in one apply. Set via
    TF_VAR_ssh_key_files='{"ncnn-sme2":"~/.ssh/mac-m4-key.pem"}'.
  EOT
  type        = map(string)
  default     = {}
}


# ---------------------------------------------------------------------------
# Mac tier — the sme2 target, and the only tier here that is not a Linux spot
# instance. EC2 Mac differs in five ways, all of which the conditionals below
# key off the instance type rather than off a separate resource, so there stays
# exactly one aws_instance / one deploy / one provisioning path:
#   1. Dedicated Host only — no shared tenancy.
#   2. No spot market.
#   3. macOS AMI, and a >= 100 GiB root volume (the AMI's own minimum).
#   4. ec2-user, not ubuntu.
#   5. No setup.sh: it is apt-based, and the AMI already ships the Apple clang
#      we compile with, so there is no cloud-init to wait on either.
# ---------------------------------------------------------------------------

locals {
  is_mac = { for label, it in var.instances : label => startswith(it, "mac") }

  name_suffix = var.namespace == "" ? "" : "-${var.namespace}"

  ssh_user = { for label, _ in var.instances : label => local.is_mac[label] ? "ec2-user" : "ubuntu" }

  # Mac labels that have a host id supplied (a label without one fails the
  # precondition on its aws_instance instead of erroring here for every label).
  mac_labels = {
    for label, it in var.instances : label => it
    if startswith(it, "mac") && contains(keys(var.mac_host_ids), label)
  }
}

# An instance has to sit in a subnet in its host's own AZ, and this VPC has a
# default subnet in all four — leave subnet_id implicit and AWS is free to pick
# a mismatched one. Read the AZ back off the supplied host and pin the
# matching default subnet.
data "aws_ec2_host" "mac" {
  for_each = local.mac_labels
  host_id  = var.mac_host_ids[each.key]
}

data "aws_subnet" "mac" {
  for_each          = data.aws_ec2_host.mac
  availability_zone = each.value.availability_zone
  default_for_az    = true
}


# ---------------------------------------------------------------------------
# Security group — SSH only
# ---------------------------------------------------------------------------

resource "aws_security_group" "kernel_testing" {
  name = "kernel-testing-sg${local.name_suffix}"

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
# Key pair — one per label. Per label so each can use its own key (Graviton
# vs Mac), and so a -target apply for one label never recreates a key pair
# other running instances were launched with. The name is stable (no
# timestamp()): a name that changes every plan makes every apply replace the
# key pair.
# ---------------------------------------------------------------------------

resource "aws_key_pair" "labeled" {
  for_each   = var.instances
  key_name   = "kernel-testing-key${local.name_suffix}-${each.key}"
  public_key = file("${trimsuffix(lookup(var.ssh_key_files, each.key, "~/.ssh/id_rsa"), ".pem")}.pub")
}

# ---------------------------------------------------------------------------
# Instances — one per label in var.instances. Replaces the old fixed
# aws_instance.kernel_testing ("c7g") / aws_instance.c8g pair: any number of
# instances, of any instance type, one per concurrent benchmarking job.
# ---------------------------------------------------------------------------

resource "aws_instance" "labeled" {
  for_each = var.instances

  dynamic "instance_market_options" {
    for_each = (var.on_demand || local.is_mac[each.key]) ? [] : [1]
    content {
      market_type = "spot"
    }
  }

  # macOS Tahoe 26 arm64 (Apple clang 21) for Mac, Ubuntu 22.04 LTS arm64 otherwise
  ami                    = local.is_mac[each.key] ? "ami-0e971f0ce976b2435" : "ami-012798e88aebdba5c"
  instance_type          = each.value
  key_name               = aws_key_pair.labeled[each.key].key_name
  vpc_security_group_ids = [aws_security_group.kernel_testing.id]

  # null for every non-Mac label, i.e. default tenancy in the default subnet.
  # Mac hosts are inputs (var.mac_host_ids), never managed here.
  tenancy   = local.is_mac[each.key] ? "host" : null
  host_id   = local.is_mac[each.key] ? lookup(var.mac_host_ids, each.key, null) : null
  subnet_id = try(data.aws_subnet.mac[each.key].id, null)

  # terminate all non-spot instance(with true shutdown but not stop the instance)
  instance_initiated_shutdown_behavior = (var.on_demand || local.is_mac[each.key]) ? "terminate" : null

  # Installs clang-18 + llvm-objdump and creates ~/arm-bench
  user_data = local.is_mac[each.key] ? null : base64encode(file("${path.module}/setup.sh"))

  root_block_device {
    # 100 GiB is the macOS AMI's own minimum; it leaves ~72 GiB free after the
    # OS and Xcode CLT, which has held every dataset we run.
    volume_size = local.is_mac[each.key] ? 100 : 50
    volume_type = "gp3"
  }

  tags = {
    Name = "kernel-testing-${each.key}"
  }

  # key_name forces a new instance. Never let a key-pair rename replace a
  # running one — for a Mac that also means re-paying the Dedicated Host's
  # 24-hour minimum.
  lifecycle {
    ignore_changes = [key_name]
    precondition {
      condition     = !local.is_mac[each.key] || contains(keys(var.mac_host_ids), each.key)
      error_message = "Mac label ${each.key} has no entry in var.mac_host_ids — eval/provision.py must supply an existing Dedicated Host id for it."
    }
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
    user        = local.ssh_user[each.key]
    private_key = file(lookup(var.ssh_key_files, each.key, "~/.ssh/id_rsa"))
    host        = aws_instance.labeled[each.key].public_ip
    timeout     = "15m"
  }

  # Block until user_data (setup.sh) is done. macOS runs no user_data and has
  # no cloud-init — its clang ships with the AMI — so assert that instead.
  provisioner "remote-exec" {
    inline = [
      local.is_mac[each.key] ? "clang --version" : "cloud-init status --wait"
    ]
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

# Which SSH user each label takes — ec2-user on Mac, ubuntu elsewhere. Read by
# eval/provision.py so the Mac/Linux split lives here and not in two places.
#
# Read off each instance's OWN instance_type, like the two outputs above.
# Computed from var.instances instead, it breaks twice over: every apply here
# is -target-scoped and terraform drops outputs that depend on nothing
# targeted, so the value vanishes from state and provision.py silently falls
# back to "ubuntu"; and state holds labels absent from var.instances, so
# indexing a var-derived map by them is an "Invalid index" error.
output "instance_ssh_users" {
  value = {
    for label, inst in aws_instance.labeled :
    label => startswith(inst.instance_type, "mac") ? "ec2-user" : "ubuntu"
  }
}

# Private key per label. Derived from the instances (not var.instances) and via
# lookup(), for the same -target / stale-state-label reasons as the output above.
output "ssh_key_paths" {
  value = {
    for label, _ in aws_instance.labeled :
    label => lookup(var.ssh_key_files, label, "~/.ssh/id_rsa")
  }
}
