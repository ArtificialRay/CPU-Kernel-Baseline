#!/bin/bash
set -e

# ---------------------------------------------------------------------------
# setup.sh — runs as root via EC2 user_data on first boot
# Installs clang-18 + llvm-objdump-18 and creates the arm-bench workspace.
# cloud-init status --wait (in the terraform null_resource) will block until
# this script exits before the rsync + build steps run.
# ---------------------------------------------------------------------------

export DEBIAN_FRONTEND=noninteractive

apt-get update -y
apt-get install -y \
    build-essential \
    make \
    cmake \
    git \
    wget \
    gnupg \
    lsb-release \
    software-properties-common \
    linux-tools-aws \
    linux-tools-common

# Install LLVM 18 from apt.llvm.org (Ubuntu 22.04 "jammy" ships clang-14 by
# default; clang-17+ is needed for SVE2 / SME2 intrinsics).
CODENAME=$(lsb_release -cs)
wget -qO /etc/apt/trusted.gpg.d/apt.llvm.org.asc https://apt.llvm.org/llvm-snapshot.gpg.key
echo "deb http://apt.llvm.org/${CODENAME}/ llvm-toolchain-${CODENAME}-18 main" \
    > /etc/apt/sources.list.d/llvm-18.list
apt-get update -y
apt-get install -y clang-18 llvm-18

# Make clang-18 / llvm-objdump-18 the system-wide defaults
update-alternatives --install /usr/bin/clang        clang        /usr/bin/clang-18        100
update-alternatives --install /usr/bin/llvm-objdump llvm-objdump /usr/bin/llvm-objdump-18 100

# Create workspace (terraform will rsync the sources here)
mkdir -p /home/ubuntu/arm-bench
chown ubuntu:ubuntu /home/ubuntu/arm-bench

echo "--- setup complete ---"
clang --version
llvm-objdump --version
