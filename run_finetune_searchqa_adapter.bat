@REM Copyright (c) Microsoft Corporation.
@REM Licensed under the MIT license.

@REM set task=entity_type
python examples/run_finetune_searchqa_adapter.py --model_type "roberta" --model_name_or_path "roberta-large" --config_name "roberta-large" --do_train --do_eval --data_dir=data/searchQA --preprocess_type read_examples_origin --output_dir=./proc_data/roberta_searchqa --max_seq_length=128 --eval_steps 200 --per_gpu_eval_batch_size=8 --gradient_accumulation_steps 8 --warmup_steps 0 --per_gpu_train_batch_size=8 --learning_rate=5e-6 --adam_epsilon 1e-6 --weight_decay 0 --train_steps 20000 --report_steps 20000000000 --freeze_bert="" --freeze_adapter="True" --adapter_size 768 --adapter_list "0,11,22" --adapter_skip_layers 0 --task_adapter "" --meta_fac_adaptermodel="./pretrained_models/fac-adapter/pytorch_model.bin" --meta_lin_adaptermodel="./pretrained_models/lin-adapter/pytorch_model.bin" --restore
