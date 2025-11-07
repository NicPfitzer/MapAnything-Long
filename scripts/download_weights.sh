#!/bin/bash
set -euo pipefail

mkdir -p weights
cd ./weights

# SALAD (~350 MiB)
echo "Downloading SALAD weights..."
SALAD_URL="https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt"
curl -L "$SALAD_URL" -o "./dino_salad.ckpt"

# DINOv2 (~340 MiB)
echo "Downloading DINO weights..."
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth

# DBoW vocabulary (~40 MiB tar.gz, ~145 MiB txt)
echo "Downloading DBoW vocabulary..."
(wget https://github.com/UZ-SLAMLab/ORB_SLAM3/raw/master/Vocabulary/ORBvoc.txt.tar.gz) & wait
tar -xzvf ORBvoc.txt.tar.gz
rm ORBvoc.txt.tar.gz

# MapAnything (~2.1 GiB)
MAPANYTHING_DIR="./mapanything"
mkdir -p "$MAPANYTHING_DIR"
MAPANYTHING_REPO="https://huggingface.co/facebook/map-anything/resolve/main"
echo "Downloading MapAnything config..."
curl -L "${MAPANYTHING_REPO}/config.json" -o "${MAPANYTHING_DIR}/config.json"
echo "Downloading MapAnything weights..."
curl -L "${MAPANYTHING_REPO}/model.safetensors" -o "${MAPANYTHING_DIR}/model.safetensors"

echo "Done. You should now have the following files in ./weights:"
echo " - dino_salad.ckpt"
echo " - dinov2_vitb14_pretrain.pth"
echo " - ORBvoc.txt"
echo " - mapanything/config.json"
echo " - mapanything/model.safetensors"
