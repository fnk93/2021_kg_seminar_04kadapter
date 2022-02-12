# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# best results for F + L
batch=64
batch=8
accu=8
lr=1e-5
GPU="0"
warmup=0
seq_length=256
fac_adap="./pretrained_models/fac-adapter/pytorch_model.bin"
lin_adap="./pretrained_models/lin-adapter/pytorch_model.bin"
comment="fl-adapter"
dataset=data/cosmosQA
CUDA_VISIBLE_DEVICES=$GPU
python examples/run_finetune_cosmosqa_adapter.py \
    --model_type roberta-large \
    --model_name_or_path roberta-large \
    --do_train \
    --comment $comment \
    --do_eval \
    --data_dir $dataset \
    --preprocess_type read_examples_origin \
    --output_dir ./proc_data \
    --max_seq_length $seq_length \
    --eval_steps 200 \
    --per_gpu_train_batch_size $batch \
    --gradient_accumulation_steps $accu \
    --warmup_steps $warmup \
    --per_gpu_eval_batch_size $batch \
    --learning_rate $lr \
    --save_steps=2000 \
    --adam_epsilon 1e-6 \
    --weight_decay 0 \
    --train_steps 20000 \
    --report_steps 20000000000 \
    --freeze_bert="" \
    --freeze_adapter="True" \
    --adapter_size 768 \
    --adapter_list "0,11,22" \
    --adapter_skip_layers 0 \
    --meta_fac_adaptermodel=$fac_adap \
    --meta_lin_adaptermodel=$lin_adap
