@REM Copyright (c) Microsoft Corporation.
@REM Licensed under the MIT license.

@REM set task=entity_type
set batch=64
set accu=8
set lr=1e-5
set GPU="0"
set warmup=0
set seq_length=256
set fac_adap="./pretrained_models/fac-adapter/pytorch_model.bin"
set lin_adap="./pretrained_models/lin-adapter/pytorch_model.bin"
set comment="fl-adapter"
set dataset="data/cosmosQA"
python examples/run_finetune_cosmosqa_adapter.py --model_type "roberta" --model_name_or_path "roberta-large" --config_name "roberta-large" --do_train --do_eval --data_dir=%dataset% --preprocess_type read_examples_origin --output_dir=./proc_data --max_seq_length=%seq_length% --eval_steps 200 --per_gpu_eval_batch_size=%batch% --gradient_accumulation_steps %accu% --warmup_steps %warmup% --per_gpu_train_batch_size=%batch% --learning_rate=%lr% --adam_epsilon 1e-6 --weight_decay 0 --train_steps 20000 --report_steps 20000000000 --freeze_bert="" --freeze_adapter="True" --adapter_size 768 --adapter_list "0,11,22" --adapter_skip_layers 0 --meta_fac_adaptermodel=%fac_adap% --meta_lin_adaptermodel=%lin_adap% --restore
