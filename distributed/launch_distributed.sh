#!/bin/bash

# 12 | 并行科技g0001     | 140.207.201.36 | ssh|sftp                                         | Linux                                          | Default                                         |
# 13 | 并行科技g0002     | 140.207.201.37 | ssh|sftp                                         | Linux                                          | Default                                         |
# 14 | 并行科技g0003     | 140.207.201.38 | ssh|sftp                                         | Linux                                          | Default                                         |
# 15 | 并行科技g0004     | 140.207.201.39 | ssh|sftp                                         | Linux                                          | Default                                         |
# 16 | 并行科技g0005     | 140.207.201.40 | ssh|sftp                                         | Linux                                          | Default                                         |
# 17 | 并行科技g0006     | 140.207.201.41 | ssh|sftp                                         | Linux                                          | Default                                         |
# 18 | 并行科技g0007     | 140.207.201.42 | ssh|sftp                                         | Linux                                          | Default                                         |
# 19 | 并行科技g0008     | 140.207.201.43 | ssh|sftp                                         | Linux                                          | Default                                         |
# 20 | 并行科技g0009     | 140.207.201.44 | ssh|sftp                                         | Linux                                          | Default                                         |
# 21 | 并行科技g0010     | 140.207.201.45 | ssh|sftp                                         | Linux                                          | Default                                         |
# 22 | 并行科技g0011     | 140.207.201.46 | ssh|sftp                                         | Linux                                          | Default                                         |
# 23 | 并行科技g0012     | 140.207.201.47 | ssh|sftp                                         | Linux                                          | Default                                         |
# 24 | 并行科技g0013     | 140.207.201.4  | ssh|sftp                                         | Linux                                          | Default                                         |
# 25 | 并行科技g0014     | 140.207.201.34 | ssh|sftp                                         | Linux                                          | Default                                         |
# 26 | 并行科技g0015     | 140.207.201.48 | ssh|sftp                                         | Linux                                          | Default                                         |
# 27 | 并行科技g0016     | 140.207.201.49 | ssh|sftp                                         | Linux                                          | Default                                         |
# 28 | 并行科技g0017     | 140.207.201.5  | ssh|sftp  

declare -A node2ip_map
node2ip_map=(
    ["g0001"]="140.207.201.36"
    ["g0002"]="140.207.201.37"
    ["g0003"]="140.207.201.38"
    ["g0004"]="140.207.201.39"
    ["g0005"]="140.207.201.40"
    ["g0006"]="140.207.201.41"
    ["g0007"]="140.207.201.42"
    ["g0008"]="140.207.201.43"
    ["g0009"]="140.207.201.44"
    ["g0010"]="140.207.201.45"
    ["g0011"]="140.207.201.46"
    ["g0012"]="140.207.201.47"
    ["g0013"]="140.207.201.4"
    ["g0014"]="140.207.201.34"
    ["g0015"]="140.207.201.48"
    ["g0016"]="140.207.201.49"
    ["g0017"]="140.207.201.5"
)

# Default nodes if no arguments provided
# DEFAULT_NODES=("g0006" "g0013" "g0014" "g0015" "g0016" "g0017")
DEFAULT_NODES=("g0014" "g0015" "g0016" "g0017")

# Local codebase path in file system
LOCAL_CODEBASE_PATH="/data1/xu_ruochen/VLM-R1"
DOCKER_WORKSPACE="/data1/xu_ruochen/VLM-R1/src/open-r1-multimodal"

# Use provided nodes or default nodes
if [ "$#" -ge 1 ]; then
    NODES=("$@")
else
    NODES=("${DEFAULT_NODES[@]}")
    echo "Using default nodes: ${NODES[*]}"
fi

# Add this debug line
echo "All nodes in order: ${NODES[@]}"

TOTAL_NODES=${#NODES[@]}
MASTER_NODE=${NODES[0]}
MASTER_PORT=12345

# Get project root directory (2 levels up from current script)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# docker compose down any running containers for all nodes
for node in "${NODES[@]}"; do
    ssh $node "sudo docker-compose -f ${PROJECT_ROOT}/scripts/distributed/docker-compose.yml down" &
    echo "Downed container on $node"
done

# wait until all docker compose down commands are finished
echo "Waiting for all docker compose down commands to finish..."
wait

# Get master node IP address
echo "MASTER_NODE: $MASTER_NODE"
MASTER_IP="${node2ip_map[$MASTER_NODE]}"
echo "Master node IP: $MASTER_IP"

# Create hostfile
echo "Generating hostfile..."
> hostfile
for node in "${NODES[@]}"; do
    echo "$node slots=8" >> hostfile
done

# Generate docker-compose.yml
echo "Generating docker-compose.yml..."
cat > docker-compose.yml << EOL
version: '3.8'

services:
  trainer:
    image: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel-r1-v
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    shm_size: '8gb'
    volumes:
      - /data1:/data1
      - /data2:/data2
      - /data3:/data3
      - /data11:/data11
      - /mnt/cfs/xu_ruochen/:/mnt/cfs/xu_ruochen/
      - /mnt/cfs/omchat_resources/:/mnt/cfs/omchat_resources/
      - $DOCKER_WORKSPACE:/workspace
    environment:
      - MASTER_ADDR=\${MASTER_ADDR:-$MASTER_IP}
      - MASTER_PORT=\${MASTER_PORT:-12345}
      - NODE_RANK=\${NODE_RANK:-0}
      - WORLD_SIZE=\${WORLD_SIZE:-4}
      - DEBUG_MODE=true
      - LOG_PATH=./log/debug_log_2b.txt
      - WANDB_API_KEY=e954056d50cf2ad0a5918d61d266b243dae84e91
      - WANDB_PROJECT=vision-reasoning
      - WANDB_RUN_NAME=Qwen-VL-2B-GRPO-$(date +%Y-%m-%d-%H-%M-%S)
      - PYTHONPATH=/workspace/src
    network_mode: "host"
    command: /bin/bash
    working_dir: $DOCKER_WORKSPACE
EOL

# Create log directory for each node
LOG_DIR="$PROJECT_ROOT/log/distributed"
mkdir -p $LOG_DIR

# Function to build training arguments from yaml
build_train_args() {
    args=""
    while IFS=": " read -r key value; do
        # Skip empty lines and comments
        [[ -z "$key" || "$key" =~ ^[[:space:]]*# ]] && continue
        
        # Remove any leading/trailing whitespace and quotes
        value=$(echo "$value" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"//' -e 's/"$//')
        
        # Handle boolean values
        if [[ "$value" == "true" ]]; then
            args="$args --$key"
        elif [[ "$value" == "false" ]]; then
            continue
        else
            args="$args --$key $value"
        fi
    done < distributed_args.yaml
    echo "$args"
}

# Get training arguments
TRAIN_ARGS=$(build_train_args)
echo "TRAIN_ARGS: $TRAIN_ARGS"

# Launch containers on each node
NODE_RANK=0
for host in "${NODES[@]}"; do
    LOG_FILE="$LOG_DIR/${host}_rank${NODE_RANK}.log"
    if [ "$host" = "$MASTER_NODE" ]; then
        echo "Launching on master $host with rank $NODE_RANK, logging to $LOG_FILE"
        ssh $host "cd $PROJECT_ROOT && \
            MASTER_ADDR=$MASTER_IP \
            NODE_RANK=$NODE_RANK \
            WORLD_SIZE=$TOTAL_NODES \
            sudo -E docker-compose -f scripts/distributed/docker-compose.yml run --rm trainer \
            torchrun --nproc_per_node=8 \
            --nnodes=$TOTAL_NODES \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_IP \
            --master_port=$MASTER_PORT \
            src/open_r1/grpo.py \
            $TRAIN_ARGS" > "$LOG_FILE" 2>&1 &
    else
        echo "Launching on $host with rank $NODE_RANK, logging to $LOG_FILE"
        ssh $host "cd $PROJECT_ROOT && \
            MASTER_ADDR=$MASTER_IP \
            NODE_RANK=$NODE_RANK \
            WORLD_SIZE=$TOTAL_NODES \
            sudo -E docker-compose -f scripts/distributed/docker-compose.yml run --rm trainer \
            torchrun --nproc_per_node=8 \
            --nnodes=$TOTAL_NODES \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_IP \
            --master_port=$MASTER_PORT \
            src/open_r1/grpo.py \
            $TRAIN_ARGS" > "$LOG_FILE" 2>&1 &
    fi
    
    NODE_RANK=$((NODE_RANK + 1))
done

echo "Jobs launched. To monitor the logs, you can:"
echo "1. Use 'tail -f $LOG_DIR/*.log' to watch all logs"
echo "2. Use 'tail -f $LOG_DIR/<node_name>_rank<N>.log' to watch a specific node"

# Wait for all background processes to complete
wait 