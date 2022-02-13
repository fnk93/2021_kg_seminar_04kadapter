# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# please use 4 GPU
task=fb15k
# GPU='0,1,2,3'
GPU='0'
# # best results for F
# seq_length=184
# batch_size=32
# lr=1e-5
# warmup=500
# fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
# lin_adap=''
# comment='f-adapter-dif-trf'
# # best results for L
# seq_length=184
# batch_size=32
# lr=1e-5
# warmup=200
# fac_adap=''
# lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
# comment='l-adapter-dif-trf'

# best results for F+L
seq_length=184
batch_size=32
lr=5e-6
warmup=1000
fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
comment='fl-fb15k-adapter-dif-trf'
dataset=data/FB15Krel
batch_size=8

CUDA_VISIBLE_DEVICES=$GPU python examples/run_finetune_TACRED_adapter.py \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --config_name roberta-large \
    --do_train  \
    --do_eval   \
    --task_name=$task     \
    --data_dir=$dataset  \
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
    --negative_sample=45000 \
    --save_steps=2000 \
    --freeze_bert="" \
    --freeze_adapter="True" \
    --adapter_size 768 \
    --adapter_list "0,11,22" \
    --adapter_skip_layers 0 \
    --meta_fac_adaptermodel=$fac_adap \
    --meta_lin_adaptermodel=$lin_adap

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# please use 4 GPU
task=fb15k
# GPU='0,1,2,3'
GPU='0'
# # best results for F
# seq_length=184
# batch_size=32
# lr=1e-5
# warmup=500
# fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
# lin_adap=''
# comment='f-adapter-dif-trf'
# best results for L
seq_length=184
batch_size=32
lr=1e-5
warmup=200
fac_adap=''
lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
comment='l-adapter-dif-trf'

# # best results for F+L
# seq_length=184
# batch_size=32
# lr=5e-6
# warmup=1000
# fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
# lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
# comment='fl-fb15k-adapter-dif-trf'
dataset=data/FB15Krel
batch_size=8

CUDA_VISIBLE_DEVICES=$GPU python examples/run_finetune_TACRED_adapter.py \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --config_name roberta-large \
    --do_train  \
    --do_eval   \
    --task_name=$task     \
    --data_dir=$dataset  \
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
    --negative_sample=45000 \
    --save_steps=2000 \
    --freeze_bert="" \
    --freeze_adapter="True" \
    --adapter_size 768 \
    --adapter_list "0,11,22" \
    --adapter_skip_layers 0 \
    --meta_fac_adaptermodel=$fac_adap \
    --meta_lin_adaptermodel=$lin_adap

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# please use 4 GPU
task=fb15k
# GPU='0,1,2,3'
GPU='0'
# best results for F
seq_length=184
batch_size=32
lr=1e-5
warmup=500
fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
lin_adap=''
comment='f-adapter-dif-trf'
# # best results for L
# seq_length=184
# batch_size=32
# lr=1e-5
# warmup=200
# fac_adap=''
# lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
# comment='l-adapter-dif-trf'

# best results for F+L
# seq_length=184
# batch_size=32
# lr=5e-6
# warmup=1000
# fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
# lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
# comment='fl-fb15k-adapter-dif-trf'
dataset=data/FB15Krel
batch_size=8

CUDA_VISIBLE_DEVICES=$GPU python examples/run_finetune_TACRED_adapter.py \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --config_name roberta-large \
    --do_train  \
    --do_eval   \
    --task_name=$task     \
    --data_dir=$dataset  \
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
    --negative_sample=45000 \
    --save_steps=2000 \
    --freeze_bert="" \
    --freeze_adapter="True" \
    --adapter_size 768 \
    --adapter_list "0,11,22" \
    --adapter_skip_layers 0 \
    --meta_fac_adaptermodel=$fac_adap \
    --meta_lin_adaptermodel=$lin_adap

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

task=entity_type_kg
# # best results on F
# seq_length=256
# batch_size=2048
# lr=5e-6
# warmup=500
# fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
# lin_adap=''
# comment='f-litwd1k-adapter'
# # best results on L
# seq_length=256
# batch_size=2048
# lr=5e-6
# warmup=200
# fac_adap=''
# lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
# comment='l-litwd1k-adapter'
# best results on F + L
seq_length=256
batch_size=2048
batch_size=16
lr=5e-6
warmup=1000
fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
comment='fl-litwd1k-adapter'
batch_size=4
dataset=data/LitWD1K
python examples/run_finetune_litWik_adapter.py \
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
    --gradient_accumulation_steps=1 \
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

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

task=entity_type_kg
# best results on F
seq_length=256
batch_size=2048
lr=5e-6
warmup=500
fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
lin_adap=''
comment='f-litwd1k-adapter'
# # best results on L
# seq_length=256
# batch_size=2048
# lr=5e-6
# warmup=200
# fac_adap=''
# lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
# comment='l-litwd1k-adapter'
# # best results on F + L
# seq_length=256
# batch_size=2048
# batch_size=16
# lr=5e-6
# warmup=1000
# fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
# lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
# comment='fl-litwd1k-adapter'
batch_size=4
dataset=data/LitWD1K
python examples/run_finetune_litWik_adapter.py \
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
    --gradient_accumulation_steps=1 \
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

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

task=entity_type_kg
# # best results on F
# seq_length=256
# batch_size=2048
# lr=5e-6
# warmup=500
# fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
# lin_adap=''
# comment='f-litwd1k-adapter'
# best results on L
seq_length=256
batch_size=2048
lr=5e-6
warmup=200
fac_adap=''
lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
comment='l-litwd1k-adapter'
# # best results on F + L
# seq_length=256
# batch_size=2048
# batch_size=16
# lr=5e-6
# warmup=1000
# fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
# lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
# comment='fl-litwd1k-adapter'
batch_size=4
dataset=data/LitWD1K
python examples/run_finetune_litWik_adapter.py \
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
    --gradient_accumulation_steps=1 \
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

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# please use 4 GPU
task=litwd
# GPU='0,1,2,3'
GPU='0'
# # best results for F
# seq_length=184
# batch_size=32
# lr=1e-5
# warmup=500
# fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
# lin_adap=''
# comment='f-adapter-dif-trf'
# # best results for L
# seq_length=184
# batch_size=32
# lr=1e-5
# warmup=200
# fac_adap=''
# lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
# comment='l-adapter-dif-trf'

# best results for F+L
seq_length=184
batch_size=32
lr=5e-6
warmup=1000
fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
comment='fl-adapter-dif-trf'
dataset=data/LitWD1Krel
batch_size=8

CUDA_VISIBLE_DEVICES=$GPU python examples/run_finetune_TACRED_adapter.py \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --config_name roberta-large \
    --do_train  \
    --do_eval   \
    --task_name=$task     \
    --data_dir=$dataset  \
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
    --negative_sample=45000 \
    --save_steps=2000 \
    --freeze_bert="" \
    --freeze_adapter="True" \
    --adapter_size 768 \
    --adapter_list "0,11,22" \
    --adapter_skip_layers 0 \
    --meta_fac_adaptermodel=$fac_adap \
    --meta_lin_adaptermodel=$lin_adap

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# please use 4 GPU
task=litwd
# GPU='0,1,2,3'
GPU='0'
# best results for F
seq_length=184
batch_size=32
lr=1e-5
warmup=500
fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
lin_adap=''
comment='f-adapter-dif-trf'
# # best results for L
# seq_length=184
# batch_size=32
# lr=1e-5
# warmup=200
# fac_adap=''
# lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
# comment='l-adapter-dif-trf'

# # best results for F+L
# seq_length=184
# batch_size=32
# lr=5e-6
# warmup=1000
# fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
# lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
# comment='fl-adapter-dif-trf'
dataset=data/LitWD1Krel
batch_size=8

CUDA_VISIBLE_DEVICES=$GPU python examples/run_finetune_TACRED_adapter.py \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --config_name roberta-large \
    --do_train  \
    --do_eval   \
    --task_name=$task     \
    --data_dir=$dataset  \
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
    --negative_sample=45000 \
    --save_steps=2000 \
    --freeze_bert="" \
    --freeze_adapter="True" \
    --adapter_size 768 \
    --adapter_list "0,11,22" \
    --adapter_skip_layers 0 \
    --meta_fac_adaptermodel=$fac_adap \
    --meta_lin_adaptermodel=$lin_adap

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# please use 4 GPU
task=litwd
# GPU='0,1,2,3'
GPU='0'
# # best results for F
# seq_length=184
# batch_size=32
# lr=1e-5
# warmup=500
# fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
# lin_adap=''
# comment='f-adapter-dif-trf'
# best results for L
seq_length=184
batch_size=32
lr=1e-5
warmup=200
fac_adap=''
lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
comment='l-adapter-dif-trf'

# # best results for F+L
# seq_length=184
# batch_size=32
# lr=5e-6
# warmup=1000
# fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
# lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
# comment='fl-adapter-dif-trf'
dataset=data/LitWD1Krel
batch_size=8

CUDA_VISIBLE_DEVICES=$GPU python examples/run_finetune_TACRED_adapter.py \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --config_name roberta-large \
    --do_train  \
    --do_eval   \
    --task_name=$task     \
    --data_dir=$dataset  \
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
    --negative_sample=45000 \
    --save_steps=2000 \
    --freeze_bert="" \
    --freeze_adapter="True" \
    --adapter_size 768 \
    --adapter_list "0,11,22" \
    --adapter_skip_layers 0 \
    --meta_fac_adaptermodel=$fac_adap \
    --meta_lin_adaptermodel=$lin_adap

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

task=entity_type_kg
# # best results on F
# seq_length=256
# batch_size=2048
# lr=5e-6
# warmup=500
# fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
# lin_adap=''
# comment='f-litwd1k-adapter'
# # best results on L
# seq_length=256
# batch_size=2048
# lr=5e-6
# warmup=200
# fac_adap=''
# lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
# comment='l-litwd1k-adapter'
# best results on F + L
seq_length=256
batch_size=2048
lr=5e-6
warmup=1000
fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
comment='fl-litwd19k-adapter'
batch_size=4
dataset=data/LitWD19K
python examples/run_finetune_litWik_adapter.py \
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
    --gradient_accumulation_steps=1 \
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

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

task=entity_type_kg
# best results on F
seq_length=256
batch_size=2048
lr=5e-6
warmup=500
fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
lin_adap=''
comment='f-litwd1k-adapter'
# # best results on L
# seq_length=256
# batch_size=2048
# lr=5e-6
# warmup=200
# fac_adap=''
# lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
# comment='l-litwd1k-adapter'
# # best results on F + L
# seq_length=256
# batch_size=2048
# lr=5e-6
# warmup=1000
# fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
# lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
# comment='fl-litwd19k-adapter'
batch_size=4
dataset=data/LitWD19K
python examples/run_finetune_litWik_adapter.py \
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
    --gradient_accumulation_steps=1 \
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

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

task=entity_type_kg
# # best results on F
# seq_length=256
# batch_size=2048
# lr=5e-6
# warmup=500
# fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
# lin_adap=''
# comment='f-litwd1k-adapter'
# best results on L
seq_length=256
batch_size=2048
lr=5e-6
warmup=200
fac_adap=''
lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
comment='l-litwd1k-adapter'
# # best results on F + L
# seq_length=256
# batch_size=2048
# lr=5e-6
# warmup=1000
# fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
# lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
# comment='fl-litwd19k-adapter'
batch_size=4
dataset=data/LitWD19K
python examples/run_finetune_litWik_adapter.py \
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
    --gradient_accumulation_steps=1 \
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

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# please use 4 GPU
task=litwd
# GPU='0,1,2,3'
GPU='0'
# # best results for F
# seq_length=184
# batch_size=32
# lr=1e-5
# warmup=500
# fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
# lin_adap=''
# comment='f-adapter-dif-trf'
# # best results for L
# seq_length=184
# batch_size=32
# lr=1e-5
# warmup=200
# fac_adap=''
# lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
# comment='l-adapter-dif-trf'

# best results for F+L
seq_length=184
batch_size=32
lr=5e-6
warmup=1000
fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
comment='fl-adapter-dif-trf'
dataset=data/LitWD19Krel
batch_size=8

CUDA_VISIBLE_DEVICES=$GPU python examples/run_finetune_TACRED_adapter.py \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --config_name roberta-large \
    --do_train  \
    --do_eval   \
    --task_name=$task     \
    --data_dir=$dataset  \
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
    --negative_sample=45000 \
    --save_steps=2000 \
    --freeze_bert="" \
    --freeze_adapter="True" \
    --adapter_size 768 \
    --adapter_list "0,11,22" \
    --adapter_skip_layers 0 \
    --meta_fac_adaptermodel=$fac_adap \
    --meta_lin_adaptermodel=$lin_adap

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# please use 4 GPU
task=litwd
# GPU='0,1,2,3'
GPU='0'
# best results for F
seq_length=184
batch_size=32
lr=1e-5
warmup=500
fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
lin_adap=''
comment='f-adapter-dif-trf'
# # best results for L
# seq_length=184
# batch_size=32
# lr=1e-5
# warmup=200
# fac_adap=''
# lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
# comment='l-adapter-dif-trf'

# # best results for F+L
# seq_length=184
# batch_size=32
# lr=5e-6
# warmup=1000
# fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
# lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
# comment='fl-adapter-dif-trf'
dataset=data/LitWD19Krel
batch_size=8

CUDA_VISIBLE_DEVICES=$GPU python examples/run_finetune_TACRED_adapter.py \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --config_name roberta-large \
    --do_train  \
    --do_eval   \
    --task_name=$task     \
    --data_dir=$dataset  \
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
    --negative_sample=45000 \
    --save_steps=2000 \
    --freeze_bert="" \
    --freeze_adapter="True" \
    --adapter_size 768 \
    --adapter_list "0,11,22" \
    --adapter_skip_layers 0 \
    --meta_fac_adaptermodel=$fac_adap \
    --meta_lin_adaptermodel=$lin_adap

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# please use 4 GPU
task=litwd
# GPU='0,1,2,3'
GPU='0'
# # best results for F
# seq_length=184
# batch_size=32
# lr=1e-5
# warmup=500
# fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
# lin_adap=''
# comment='f-adapter-dif-trf'
# best results for L
seq_length=184
batch_size=32
lr=1e-5
warmup=200
fac_adap=''
lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
comment='l-adapter-dif-trf'

# # best results for F+L
# seq_length=184
# batch_size=32
# lr=5e-6
# warmup=1000
# fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
# lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
# comment='fl-adapter-dif-trf'
dataset=data/LitWD19Krel
batch_size=8

CUDA_VISIBLE_DEVICES=$GPU python examples/run_finetune_TACRED_adapter.py \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --config_name roberta-large \
    --do_train  \
    --do_eval   \
    --task_name=$task     \
    --data_dir=$dataset  \
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
    --negative_sample=45000 \
    --save_steps=2000 \
    --freeze_bert="" \
    --freeze_adapter="True" \
    --adapter_size 768 \
    --adapter_list "0,11,22" \
    --adapter_skip_layers 0 \
    --meta_fac_adaptermodel=$fac_adap \
    --meta_lin_adaptermodel=$lin_adap

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

task=entity_type_kg
# # best results on F
# seq_length=256
# batch_size=2048
# lr=5e-6
# warmup=500
# fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
# lin_adap=''
# comment='f-litwd1k-adapter'
# # best results on L
# seq_length=256
# batch_size=2048
# lr=5e-6
# warmup=200
# fac_adap=''
# lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
# comment='l-litwd1k-adapter'
# best results on F + L
seq_length=256
batch_size=2048
batch_size=4
lr=5e-6
warmup=1000
fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
comment='fl-litwd48k-adapter'
dataset=data/LitWD48K
python examples/run_finetune_litWik_adapter.py \
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
    --gradient_accumulation_steps=1 \
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

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

task=entity_type_kg
# best results on F
seq_length=256
batch_size=2048
lr=5e-6
warmup=500
fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
lin_adap=''
comment='f-litwd1k-adapter'
# # best results on L
# seq_length=256
# batch_size=2048
# lr=5e-6
# warmup=200
# fac_adap=''
# lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
# comment='l-litwd1k-adapter'
# # best results on F + L
# seq_length=256
# batch_size=2048
# lr=5e-6
# warmup=1000
# fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
# lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
# comment='fl-litwd48k-adapter'
batch_size=4
dataset=data/LitWD48K
python examples/run_finetune_litWik_adapter.py \
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
    --gradient_accumulation_steps=1 \
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

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

task=entity_type_kg
# # best results on F
# seq_length=256
# batch_size=2048
# lr=5e-6
# warmup=500
# fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
# lin_adap=''
# comment='f-litwd1k-adapter'
# best results on L
seq_length=256
batch_size=2048
lr=5e-6
warmup=200
fac_adap=''
lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
comment='l-litwd1k-adapter'
# # best results on F + L
# seq_length=256
# batch_size=2048
# lr=5e-6
# warmup=1000
# fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
# lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
# comment='fl-litwd48k-adapter'
batch_size=4
dataset=data/LitWD48K
python examples/run_finetune_litWik_adapter.py \
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
    --gradient_accumulation_steps=1 \
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

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# please use 4 GPU
task=litwd
# GPU='0,1,2,3'
GPU='0'
# # best results for F
# seq_length=184
# batch_size=32
# lr=1e-5
# warmup=500
# fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
# lin_adap=''
# comment='f-adapter-dif-trf'
# # best results for L
# seq_length=184
# batch_size=32
# lr=1e-5
# warmup=200
# fac_adap=''
# lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
# comment='l-adapter-dif-trf'

# best results for F+L
seq_length=184
batch_size=32
lr=5e-6
warmup=1000
fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
comment='fl-adapter-dif-trf'
dataset=data/LitWD48Krel
batch_size=8

CUDA_VISIBLE_DEVICES=$GPU python examples/run_finetune_TACRED_adapter.py \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --config_name roberta-large \
    --do_train  \
    --do_eval   \
    --task_name=$task     \
    --data_dir=$dataset  \
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
    --negative_sample=45000 \
    --save_steps=2000 \
    --freeze_bert="" \
    --freeze_adapter="True" \
    --adapter_size 768 \
    --adapter_list "0,11,22" \
    --adapter_skip_layers 0 \
    --meta_fac_adaptermodel=$fac_adap \
    --meta_lin_adaptermodel=$lin_adap

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# please use 4 GPU
task=litwd
# GPU='0,1,2,3'
GPU='0'
# best results for F
seq_length=184
batch_size=32
lr=1e-5
warmup=500
fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
lin_adap=''
comment='f-adapter-dif-trf'
# # best results for L
# seq_length=184
# batch_size=32
# lr=1e-5
# warmup=200
# fac_adap=''
# lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
# comment='l-adapter-dif-trf'

# # best results for F+L
# seq_length=184
# batch_size=32
# lr=5e-6
# warmup=1000
# fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
# lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
# comment='fl-adapter-dif-trf'
dataset=data/LitWD48Krel
batch_size=8

CUDA_VISIBLE_DEVICES=$GPU python examples/run_finetune_TACRED_adapter.py \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --config_name roberta-large \
    --do_train  \
    --do_eval   \
    --task_name=$task     \
    --data_dir=$dataset  \
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
    --negative_sample=45000 \
    --save_steps=2000 \
    --freeze_bert="" \
    --freeze_adapter="True" \
    --adapter_size 768 \
    --adapter_list "0,11,22" \
    --adapter_skip_layers 0 \
    --meta_fac_adaptermodel=$fac_adap \
    --meta_lin_adaptermodel=$lin_adap

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# please use 4 GPU
task=litwd
# GPU='0,1,2,3'
GPU='0'
# # best results for F
# seq_length=184
# batch_size=32
# lr=1e-5
# warmup=500
# fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
# lin_adap=''
# comment='f-adapter-dif-trf'
# best results for L
seq_length=184
batch_size=32
lr=1e-5
warmup=200
fac_adap=''
lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
comment='l-adapter-dif-trf'

# # best results for F+L
# seq_length=184
# batch_size=32
# lr=5e-6
# warmup=1000
# fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
# lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
# comment='fl-adapter-dif-trf'
dataset=data/LitWD48Krel
batch_size=8

CUDA_VISIBLE_DEVICES=$GPU python examples/run_finetune_TACRED_adapter.py \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --config_name roberta-large \
    --do_train  \
    --do_eval   \
    --task_name=$task     \
    --data_dir=$dataset  \
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
    --negative_sample=45000 \
    --save_steps=2000 \
    --freeze_bert="" \
    --freeze_adapter="True" \
    --adapter_size 768 \
    --adapter_list "0,11,22" \
    --adapter_skip_layers 0 \
    --meta_fac_adaptermodel=$fac_adap \
    --meta_lin_adaptermodel=$lin_adap

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# please use 4 GPU
task=wn18rr
# GPU='0,1,2,3'
GPU='0'
# # best results for F
# seq_length=184
# batch_size=32
# lr=1e-5
# warmup=500
# fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
# lin_adap=''
# comment='f-adapter-dif-trf'
# # best results for L
# seq_length=184
# batch_size=32
# lr=1e-5
# warmup=200
# fac_adap=''
# lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
# comment='l-adapter-dif-trf'

# best results for F+L
seq_length=184
batch_size=32
lr=5e-6
warmup=1000
fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
comment='fl-wn18rr-adapter-dif-trf'
dataset=data/WN18RRrel
batch_size=8

CUDA_VISIBLE_DEVICES=$GPU python examples/run_finetune_TACRED_adapter.py \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --config_name roberta-large \
    --do_train  \
    --do_eval   \
    --task_name=$task     \
    --data_dir=$dataset  \
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
    --negative_sample=45000 \
    --save_steps=2000 \
    --freeze_bert="" \
    --freeze_adapter="True" \
    --adapter_size 768 \
    --adapter_list "0,11,22" \
    --adapter_skip_layers 0 \
    --meta_fac_adaptermodel=$fac_adap \
    --meta_lin_adaptermodel=$lin_adap

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# please use 4 GPU
task=wn18rr
# GPU='0,1,2,3'
GPU='0'
# best results for F
seq_length=184
batch_size=32
lr=1e-5
warmup=500
fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
lin_adap=''
comment='f-adapter-dif-trf'
# # best results for L
# seq_length=184
# batch_size=32
# lr=1e-5
# warmup=200
# fac_adap=''
# lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
# comment='l-adapter-dif-trf'

# # best results for F+L
# seq_length=184
# batch_size=32
# lr=5e-6
# warmup=1000
# fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
# lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
# comment='fl-wn18rr-adapter-dif-trf'
dataset=data/WN18RRrel
batch_size=8

CUDA_VISIBLE_DEVICES=$GPU python examples/run_finetune_TACRED_adapter.py \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --config_name roberta-large \
    --do_train  \
    --do_eval   \
    --task_name=$task     \
    --data_dir=$dataset  \
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
    --negative_sample=45000 \
    --save_steps=2000 \
    --freeze_bert="" \
    --freeze_adapter="True" \
    --adapter_size 768 \
    --adapter_list "0,11,22" \
    --adapter_skip_layers 0 \
    --meta_fac_adaptermodel=$fac_adap \
    --meta_lin_adaptermodel=$lin_adap

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# please use 4 GPU
task=wn18rr
# GPU='0,1,2,3'
GPU='0'
# # best results for F
# seq_length=184
# batch_size=32
# lr=1e-5
# warmup=500
# fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
# lin_adap=''
# comment='f-adapter-dif-trf'
# best results for L
seq_length=184
batch_size=32
lr=1e-5
warmup=200
fac_adap=''
lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
comment='l-adapter-dif-trf'

# # best results for F+L
# seq_length=184
# batch_size=32
# lr=5e-6
# warmup=1000
# fac_adap='./pretrained_models/fac-adapter/pytorch_model.bin'
# lin_adap='./pretrained_models/lin-adapter/pytorch_model.bin'
# comment='fl-wn18rr-adapter-dif-trf'
dataset=data/WN18RRrel
batch_size=8

CUDA_VISIBLE_DEVICES=$GPU python examples/run_finetune_TACRED_adapter.py \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --config_name roberta-large \
    --do_train  \
    --do_eval   \
    --task_name=$task     \
    --data_dir=$dataset  \
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
    --negative_sample=45000 \
    --save_steps=2000 \
    --freeze_bert="" \
    --freeze_adapter="True" \
    --adapter_size 768 \
    --adapter_list "0,11,22" \
    --adapter_skip_layers 0 \
    --meta_fac_adaptermodel=$fac_adap \
    --meta_lin_adaptermodel=$lin_adap
