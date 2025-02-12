# Distributed Training Launch Script

This directory contains scripts for launching distributed training across multiple nodes using Docker containers.

## Overview

The `launch_distributed.sh` script automates the process of setting up and launching distributed PyTorch training across multiple nodes in a cluster environment.

## Prerequisites

- SSH access to all nodes
- Docker and docker-compose installed on all nodes
- NVIDIA GPUs and nvidia-docker runtime
- PyTorch 2.5.1 or compatible version

## Node Configuration

The script maintains a mapping of node names to IP addresses. Available nodes:
- g0001 - g0017 (See node2ip_map in launch_distributed.sh for complete mapping)

Default nodes used if none specified:
- g0014, g0015, g0016, g0017

## Usage

### Basic Usage
