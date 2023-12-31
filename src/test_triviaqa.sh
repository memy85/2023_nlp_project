#!/bin/bash

python run_qa.py \
  --model_name_or_path gpt2 \
  --dataset_name trivia_qa \
  --dataset_config_name unfiltered \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/qa_trivia