@ECHO OFF
CLS

REM CHANGE THESE SETTINGS ACCORDING TO THE MODEL:
SET TYPE=TF-S
SET MODEL=SEBIS/code_trans_t5_small_code_documentation_generation_go_transfer_learning_finetune

REM MAY BE ONE OF THESE:
REM TF-S SEBIS/code_trans_t5_small_code_documentation_generation_go_transfer_learning_finetune
REM TF-B SEBIS/code_trans_t5_base_code_documentation_generation_go_transfer_learning_finetune
REM TF-L SEBIS/code_trans_t5_large_code_documentation_generation_go_transfer_learning_finetune
REM MT-TF-S SEBIS/code_trans_t5_small_code_documentation_generation_go_multitask
REM MT-TF-B SEBIS/code_trans_t5_base_code_documentation_generation_go_multitask
REM MT-TF-L SEBIS/code_trans_t5_large_code_documentation_generation_go_multitask

REM SET LANGUAGE OF THE TRAINING DATASET:
SET LANGUAGE=go

REM ==================
REM DO NOT CHANGE THIS
SET OUTPUT_DIR=models/%TYPE%_%LANGUAGE%_%DATE%
SET TEXTCOLUMN=code
SET SUMMARYCOLUMN=nl


python main.py ^
    --model_name_or_path %MODEL% ^
    --do_train ^
    --train_file datasets/%LANGUAGE%/train.json ^
    --test_file datasets/%LANGUAGE%/test.json ^
    --text_column %TEXTCOLUMN% ^
    --summary_column %SUMMARYCOLUMN% ^
    --optim adamw_torch ^
    --source_prefix "summarize: " ^
    --output_dir %OUTPUT_DIR% ^
    --per_device_train_batch_size=4 ^
    --per_device_eval_batch_size=4 ^
    --overwrite_output_dir ^
    --use_fast_tokenizer=False ^
    --predict_with_generate

