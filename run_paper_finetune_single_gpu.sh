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

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

task=entity_type
# best results for F
seq_length=256
lr=5e-6
warmup=500
batch_size=4
fac_adap="./pretrained_models/fac-adapter/pytorch_model.bin"
lin_adap=""
comment='f-adapter-trf'
# # best results for L
# seq_length=256
# lr=5e-6
# warmup=1000
# batch_size=4
# fac_adap=""
# lin_adap="./pretrained_models/lin-adapter/pytorch_model.bin"
# comment='l-adapter-trf'
# # best results for F + L
# seq_length=256
# lr=5e-6
# warmup=1000
# batch_size=4
# fac_adap="./pretrained_models/fac-adapter/pytorch_model.bin"
# lin_adap="./pretrained_models/lin-adapter/pytorch_model.bin"
# comment='fl-adapter-trf'

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
# best results for L
seq_length=256
lr=5e-6
warmup=1000
batch_size=4
fac_adap=""
lin_adap="./pretrained_models/lin-adapter/pytorch_model.bin"
comment='l-adapter-trf'
# # best results for F + L
# seq_length=256
# lr=5e-6
# warmup=1000
# batch_size=4
# fac_adap="./pretrained_models/fac-adapter/pytorch_model.bin"
# lin_adap="./pretrained_models/lin-adapter/pytorch_model.bin"
# comment='fl-adapter-trf'

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

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# please use 4 GPU
task=tacred
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
dataset=data/TACRED
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
task=tacred
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
# dataset=data/TACRED
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
task=tacred
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
# dataset=data/TACRED
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
