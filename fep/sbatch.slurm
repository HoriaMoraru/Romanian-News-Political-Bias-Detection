#!/bin/bash
#SBATCH --export=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --partition=dgxa100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=16G
#SBATCH --time=11:59:59
#SBATCH --account=student
#SBATCH --job-name=detect-ner
#SBATCH --output=logs2/train-%j.out
#SBATCH --error=logs2/train-%j.err

apptainer exec --nv \
  --bind /export/home/acs/stud/s/stefania.silivestru/models:/models \
  fep/sandbox \
  bash -c '
    source /vllm-venv/bin/activate

    vllm --version
    echo "[INFO] Starting vLLM server..."
    vllm serve \
      /models/Llama-3.3-70B-Instruct-bnb-4bit \
      --quantization bitsandbytes \
      --load-format bitsandbytes \
      --trust-remote-code \
      --gpu-memory-utilization 0.8 \
      --swap-space 8 \
      --tokenizer /models/Llama-3.3-70B-Instruct-bnb-4bit \
      --max-model-len 8000 \
      --dtype float16 \
      --port 8000 &

    echo "[INFO] Waiting for vLLM server to be ready..."
    until curl -s http://localhost:8000/v1/models > /dev/null; do
      echo "[INFO] Still waiting for vLLM to start..."
      sleep 5
    done

    curl http://localhost:8000/v1/models

    echo "[INFO] vLLM server is up. Running inference script..."
    python3 llm_extract_text_sentiment.py 1
'

