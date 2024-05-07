export DATA_PATH=../dataset/no_board_history_with_sys_history_v2.json
export CKPT_PATH=meta-llama/Llama-2-7b-chat-hf
export SAVE_PATH=../output/no_board_history_with_sys_history_v2_10epoch_lr5e-5_batch2


python sft.py \
    --dataset_name=${DATA_PATH}\
    --model_name_or_path=${CKPT_PATH} \
    --report_to="wandb" \
    --learning_rate=5e-5 \
    --per_device_train_batch_size=2 \
    --gradient_accumulation_steps=16 \
    --output_dir=${SAVE_PATH} \
    --logging_steps=1 \
    --num_train_epochs=10 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing \
    --use_peft \
    --lora_r=16 \
    --lora_alpha=32 \
    --lora_dropout=0.05 \
    --lora_target_modules "q_proj" "v_proj"\
    --torch_dtype=bfloat16 \
    --bf16=True
    
    