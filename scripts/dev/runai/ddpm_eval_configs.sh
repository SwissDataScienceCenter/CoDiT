#!/bin/bash
# Usage: ./ddpm_eval_configs.sh <dataset> <config_name> <split>

dataset=$1
config=$2
split=$3

source ~/.bashrc
micromamba activate codit
cd /mydata/adopt/christian/CoDiT/scripts/user/eval

if [[ "$dataset" == "open_ocean" ]]; then
    pred_len=72
elif [[ "$dataset" == "coastline" ]]; then
    pred_len=24
else
    echo "Unknown dataset: $dataset" >&2
    exit 1
fi

# Default values
model_type=""
optimizer=""
scheduler=""
context_encoder_type=""

# Configurations
case $config in
  transformer1)
    model_type="Transformer"
    optimizer="adam"
    scheduler="constant"
    context_encoder_type="conv"
    K=100
    input_vars="hycom_current copernicus_current era5_wind bathymetry"
    ;;
  transformer2)
    model_type="Transformer"
    optimizer="adam"
    scheduler="constant"
    context_encoder_type="conv"
    K=100
    input_vars="hycom_current copernicus_current era5_wind"
    ;;
  transformer3)
    model_type="Transformer"
    optimizer="adam"
    scheduler="constant"
    context_encoder_type="conv"
    K=100
    input_vars="hycom_current copernicus_current"
    ;;
  transformer4)
    model_type="Transformer"
    optimizer="adam"
    scheduler="constant"
    context_encoder_type="conv"
    K=100
    input_vars="hycom_current"
    ;;
  transformer5)
    model_type="Transformer"
    optimizer="adam"
    scheduler="constant"
    context_encoder_type="conv"
    K=100
    input_vars="copernicus_current"
    ;;
  transformer6)
    model_type="Transformer"
    optimizer="adam"
    scheduler="constant"
    context_encoder_type="conv"
    K=100
    input_vars="copernicus_current era5_wind bathymetry"
    ;;
  gru1)
    model_type="GRU"
    optimizer="adam"
    scheduler="constant"
    context_encoder_type="conv"
    K=100
    input_vars="hycom_current copernicus_current era5_wind bathymetry"
    ;;
  gru2)
    model_type="GRU"
    optimizer="adam"
    scheduler="constant"
    context_encoder_type="conv"
    K=100
    input_vars="hycom_current copernicus_current era5_wind"
    ;;
  gru3)
    model_type="GRU"
    optimizer="adam"
    scheduler="constant"
    context_encoder_type="conv"
    K=100
    input_vars="hycom_current copernicus_current"
    ;;
  gru4)
    model_type="GRU"
    optimizer="adam"
    scheduler="constant"
    context_encoder_type="conv"
    K=100
    input_vars="hycom_current"
    ;;
  gru5)
    model_type="GRU"
    optimizer="adam"
    scheduler="constant"
    context_encoder_type="conv"
    K=100
    input_vars="copernicus_current"
    ;;
  gru6)
    model_type="GRU"
    optimizer="adam"
    scheduler="constant"
    context_encoder_type="conv"
    K=100
    input_vars="copernicus_current era5_wind bathymetry"
    ;;
  * )
    echo "Unknown config: $config"
    exit 1
    ;;
esac

python codit_eval.py \
  --dataset "$dataset" \
  --split "$split" \
  --model_type "$model_type" \
  --context_encoder_type "$context_encoder_type" \
  --optimizer "$optimizer" \
  --learning_rate 1e-3 \
  --scheduler "$scheduler" \
  --K "$K" \
  --input_vars $input_vars \
  --prediction_length "$pred_len"

