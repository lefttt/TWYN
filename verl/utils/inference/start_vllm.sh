set -x


vllm serve ${MODEL_PATH}/huggingface \
    --port 8000 \
    --max-model-len 32768 \
    --served-model-name "qwen-2.5-instruct_1.5b" \
    --trust-remote-code \
    --disable-log-requests

