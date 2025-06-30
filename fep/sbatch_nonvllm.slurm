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

# Save the first command-line argument (the URL)
URL="$1"

apptainer exec --nv \
  --bind /export/home/acs/stud/s/stefania.silivestru/horia/RomanianNewsPoliticalBiasDetection/.env:/app/.env \
  fep/sandbox \
  bash -c "
    source /demo_venv/bin/activate

    echo '[INFO] Running script with URL: $URL'
    python3 demo.py \"$URL\"
  "
