@REM Copyright (c) Microsoft Corporation.
@REM Licensed under the MIT license.

set task=entity_type_kg
set warmup=120
set learning=3e-5
set batch_size=4
set seq_length=256
set dataset=data/LitWD1K
python examples/run_finetune_figer_adapter.py --model_type "roberta" --model_name_or_path "roberta-large" --config_name "roberta-large" --do_train --evaluate_during_training --do_eval --task_name=%task% --data_dir=%dataset% --output_dir=./proc_data --comment 'figer-adapter' --max_seq_length=%seq_length% --per_gpu_eval_batch_size=%batch_size% --per_gpu_train_batch_size=%batch_size% --learning_rate=%learning% --gradient_accumulation_steps=1 --max_steps=-1 --model_name=roberta-large --overwrite_output_dir --overwrite_cache --warmup_steps=%warmup% --save_steps=100 --freeze_bert="" --freeze_adapter="True" --adapter_size 768 --adapter_list "0,11,22" --adapter_skip_layers 0 --task_adapter "" --meta_fac_adaptermodel="./pretrained_models/fac-adapter/pytorch_model.bin" --meta_lin_adaptermodel="./pretrained_models/lin-adapter/pytorch_model.bin" --restore
