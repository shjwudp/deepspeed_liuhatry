#!/bin/bash


DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs

# export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

BASE_DATA_PATH=/share/ai_platform/changjianbin/playground/Megatron-DeepSpeed/
DATASET=${BASE_DATA_PATH}/my-gpt2_text_document
VOCAB_PATH=${BASE_DATA_PATH}/gpt2-vocab.json
MERGE_PATH=${BASE_DATA_PATH}/gpt2-merges.txt

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
CONFIG_JSON="$script_dir/ds_config.json"

USE_DEEPSPEED=1
ZERO_STAGE=0
HOSTFILE=""


# test
PROJECT="4layers_deepspeed_1node"
TP=1
PP=1
LAYERS=4
MICRO_BATCH=32
GLOBAL_BATCH=4096
MOE="deepspeed"
NUM_EXPERTS=40
TOP_K=1

HIDDEN=3072
NUM_ATTN_HEADS=48
EXPERT_HIDDEN=6144
SEQ=128
WORKER_STR=""

TIME=$(date +%Y-%m-%d-%H-%M-%S)
MODEL_NAME=layers${LAYERS}_pipline${PP}_micro${MICRO_BATCH}_global${GLOBAL_BATCH}_moe${MOE}_topk${TOP_K}_experts${NUM_EXPERTS}_zero${ZERO_STAGE}_${TIME}
TENSORBOARD_DIR=./tensorboard/${PROJECT}/${MODEL_NAME}
LOGS_DIR="logs/"${PROJECT}
mkdir -p $LOGS_DIR

while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    --no-deepspeed)
    USE_DEEPSPEED=0;
    shift
    ;;
    -z|--zero-stage)
    ZERO_STAGE=$2;
    shift
    ;;
    *)
    echo "Unknown argument(s)"
    usage
    exit 1
    shift
    ;;
esac
done


options=" \
        --moe ${MOE} \
        --use-tutel \
        --tensor-model-parallel-size $TP \
        --pipeline-model-parallel-size $PP \
        --num-layers $LAYERS \
        --hidden-size $HIDDEN \
        --expert-hidden-size $EXPERT_HIDDEN \
        --num-experts $NUM_EXPERTS \
        --top-k ${TOP_K} \
        --num-attention-heads $NUM_ATTN_HEADS \
        --seq-length $SEQ \
        --loss-scale 12 \
        --max-position-embeddings 512 \
        --micro-batch-size $MICRO_BATCH \
        --global-batch-size $GLOBAL_BATCH \
        --train-iters 1000000 \
        --lr 6.0e-5 \
        --min-lr 6.0e-6 \
        --lr-decay-style cosine \
        --log-interval 10 \
        --eval-iters 40 \
        --eval-interval 1000 \
        --data-path ${DATASET} \
        --vocab-file ${VOCAB_PATH} \
        --merge-file ${MERGE_PATH} \
        --save-interval 1000000 \
        --split 98,2,0 \
        --clip-grad 1.0 \
        --weight-decay 0.1 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --init-method-std 0.006 \
        --fp16 \
        --checkpoint-activations \
        --log-num-zeros-in-grad \
        --log-params-norm \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensorboard-dir ${TENSORBOARD_DIR}
        "


if [[ ${USE_DEEPSPEED} -eq 1 ]]; then
	echo "Using DeepSpeed"
	options="${options} \
		--deepspeed \
		--deepspeed_config=${CONFIG_JSON} \
		--zero-stage=${ZERO_STAGE} \
		--deepspeed-activation-checkpointing \
	"
fi


cat <<EOT > $CONFIG_JSON
{
  "train_batch_size" : $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": 10,

  "zero_optimization": {
    "stage": $ZERO_STAGE,
    "contiguous_gradients": true
  },

  "gradient_clipping": 1.0,
  "prescale_gradients": false,

  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 500,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },

  "wall_clock_breakdown" : true
}
EOT


#   "zero_optimization": {
#     "stage": [0|1|2],
#     "stage3_max_live_parameters" : 1000000000,
#     "stage3_max_reuse_distance" : 1000000000,
#     "allgather_partitions": [true|false],
#     "allgather_bucket_size": 500000000,
#     "reduce_scatter": [true|false],
#     "contiguous_gradients" : [true|false]
#     "overlap_comm": [true|false],
#     "reduce_bucket_size": 500000000,
#     "load_from_fp32_weights": [true|false],
#     "cpu_offload": [true|false] (deprecated),
#     "cpu_offload_params" : [true|false] (deprecated),
#     "cpu_offload_use_pin_memory": [true|false] (deprecated),
#     "sub_group_size" : 1000000000000,
#     "offload_param": {...},
#     "offload_optimizer": {...},
#     "ignore_unused_parameters": [true|false],
#     "round_robin_gradients": [true|false]
#   }

#   "zero_optimization": {
#     "stage": 1,
#   },

run_cmd="nohup deepspeed --hostfile ./${HOSTFILE} $WORKER_STR ${DIR}/pretrain_gpt.py $@ ${options} > ${LOGS_DIR}/${MODEL_NAME}.log 2>&1 &"

echo ${run_cmd}
eval ${run_cmd}

set +x
