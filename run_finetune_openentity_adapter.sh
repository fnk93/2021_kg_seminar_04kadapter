# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

task=entity_type
# # best results for F
# seq_length=256
# lr=5e-6
# warmup=500
# batch_size=4
# fac_adap="./pretrained_models/fac-adapter/pytorch_model.bin"
# lin_adap=""
# comment='f-adapter-trf'
# # best results for L
# seq_length=256
# lr=5e-6
# warmup=1000
# batch_size=4
# fac_adap=""
# lin_adap="./pretrained_models/lin-adapter/pytorch_model.bin"
# comment='l-adapter-trf'
# best results for F + L
seq_length=256
lr=5e-6
warmup=1000
batch_size=4
fac_adap="./pretrained_models/fac-adapter/pytorch_model.bin"
lin_adap="./pretrained_models/lin-adapter/pytorch_model.bin"
comment='fl-adapter-trf'

dataset=data/OpenEntity

python examples/run_finetune_openentity_adapter.py \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --config_name roberta-large \
    --do_train  \
    --do_eval   \
    --task_name=$task     \
    --data_dir=$dataset \
    --output_dir=./proc_data  \
    --comment $comment \
    --max_seq_length=$seq_length  \
    --per_gpu_eval_batch_size=$batch_size   \
    --per_gpu_train_batch_size=$batch_size   \
    --learning_rate=$lr \
    --gradient_accumulation_steps=1 \
    --max_steps=12000  \
    --model_name=roberta-large  \
    --overwrite_output_dir   \
    --overwrite_cache \
    --warmup_steps=$warmup \
    --save_steps=2000 \
    --freeze_bert="" \
    --freeze_adapter="True" \
    --adapter_size 768 \
    --adapter_list "0,11,22" \
    --adapter_skip_layers 0 \
    --meta_fac_adaptermodel=$fac_adap \
    --meta_lin_adaptermodel=$lin_adap
