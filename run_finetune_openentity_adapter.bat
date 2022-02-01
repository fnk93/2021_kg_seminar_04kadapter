@REM Copyright (c) Microsoft Corporation.
@REM Licensed under the MIT license.

set task=entity_type
set batch_size=4
set learning=1e-4
set warmup=120
@REM echo %learning%
python examples/run_finetune_openentity_adapter.py --data_dir=data/OpenEntity --model_type "roberta" --model_name_or_path "roberta-large" --config_name "roberta-large" --do_train --do_eval --task_name=%task% --output_dir=./proc_data --comment 'combine-adapter-trf' --max_seq_length=256 --per_gpu_eval_batch_size=%batch_size% --per_gpu_train_batch_size=%batch_size% --learning_rate=%learning% --gradient_accumulation_steps=1 --max_steps=12000 --model_name=roberta-large --overwrite_output_dir --overwrite_cache --warmup_steps=%warmup% --save_steps=1000 --freeze_bert="" --freeze_adapter="True" --adapter_size 768 --adapter_list "0,11,22" --adapter_skip_layers 0 --meta_fac_adaptermodel="./pretrained_models/fac-adapter/pytorch_model.bin" --meta_lin_adaptermodel="./pretrained_models/lin-adapter/pytorch_model.bin" --restore
