# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

task=entity_type
# # best results on F
# seq_length=256
# batch_size=2048
# lr=5e-6
# warmup=500
# fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
# lin_adap=''
# comment='f-figer-adapter'
# # best results on L
# seq_length=256
# batch_size=2048
# lr=5e-6
# warmup=200
# fac_adap=''
# lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
# comment='l-figer-adapter'
# best results on F + L
seq_length=256
batch_size=2048
lr=5e-6
warmup=1000
fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
comment='fl-figer-adapter'
batch_size=4
accu=512

dataset=data/FIGER

python examples/run_finetune_figer_adapter.py \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --config_name roberta-large \
    --do_train  \
    --evaluate_during_training \
    --do_eval   \
    --task_name=$task     \
    --data_dir=$dataset  \
    --output_dir=./proc_data \
    --comment $comment \
    --max_seq_length=$seq_length  \
    --per_gpu_eval_batch_size=$batch_size   \
    --per_gpu_train_batch_size=$batch_size   \
    --learning_rate=$lr \
    --gradient_accumulation_steps=$accu \
    --max_steps=-1  \
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
    --task_adapter '' \
    --meta_fac_adaptermodel=$fac_adap \
    --meta_lin_adaptermodel=$lin_adap \
    --save_to_s3 \
    --read_from_s3 \
    --num_train_epochs=3


# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

task=entity_type
# best results on F
seq_length=256
batch_size=2048
lr=5e-6
warmup=500
fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
lin_adap=''
comment='f-figer-adapter'
# # best results on L
# seq_length=256
# batch_size=2048
# lr=5e-6
# warmup=200
# fac_adap=''
# lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
# comment='l-figer-adapter'
# best results on F + L
# seq_length=256
# batch_size=2048
# lr=5e-6
# warmup=1000
# fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
# lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
# comment='fl-figer-adapter'
batch_size=4

dataset=data/FIGER

python examples/run_finetune_figer_adapter.py \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --config_name roberta-large \
    --do_train  \
    --evaluate_during_training \
    --do_eval   \
    --task_name=$task     \
    --data_dir=$dataset  \
    --output_dir=./proc_data \
    --comment $comment \
    --max_seq_length=$seq_length  \
    --per_gpu_eval_batch_size=$batch_size   \
    --per_gpu_train_batch_size=$batch_size   \
    --learning_rate=$lr \
    --gradient_accumulation_steps=$accu \
    --max_steps=-1  \
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
    --task_adapter '' \
    --meta_fac_adaptermodel=$fac_adap \
    --meta_lin_adaptermodel=$lin_adap \
    --save_to_s3 \
    --read_from_s3 \
    --num_train_epochs=3


# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

task=entity_type
# # best results on F
# seq_length=256
# batch_size=2048
# lr=5e-6
# warmup=500
# fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
# lin_adap=''
# comment='f-figer-adapter'
# best results on L
seq_length=256
batch_size=2048
lr=5e-6
warmup=200
fac_adap=''
lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
comment='l-figer-adapter'
# best results on F + L
# seq_length=256
# batch_size=2048
# lr=5e-6
# warmup=1000
# fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
# lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
# comment='fl-figer-adapter'
batch_size=4

dataset=data/FIGER

python examples/run_finetune_figer_adapter.py \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --config_name roberta-large \
    --do_train  \
    --evaluate_during_training \
    --do_eval   \
    --task_name=$task     \
    --data_dir=$dataset  \
    --output_dir=./proc_data \
    --comment $comment \
    --max_seq_length=$seq_length  \
    --per_gpu_eval_batch_size=$batch_size   \
    --per_gpu_train_batch_size=$batch_size   \
    --learning_rate=$lr \
    --gradient_accumulation_steps=$accu \
    --max_steps=-1  \
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
    --task_adapter '' \
    --meta_fac_adaptermodel=$fac_adap \
    --meta_lin_adaptermodel=$lin_adap \
    --save_to_s3 \
    --read_from_s3 \
    --num_train_epochs=3
