#!/bin/bash
LC_NUMERIC=POSIX


bs=32
epochs=5.0
lr="0.0002"
echo $1

for task in cola mnli mrpc qnli qqp rte sst2 stsb wnli;
do
    echo $bs
    echo $task
    #mkdir "../../../../../../../raid/damien/glue/outputs/$task_$bs_$1"
    CUDA_VISIBLE_DEVICES=1,2,3 python run_glue.py --model_name_or_path bert-base-uncased --task_name $task   --do_train   --do_eval  --max_seq_length 128   --per_gpu_train_batch_size $bs  --output_dir "../../../../../../../raid/damien/glue/outputs/${task}_${bs}_cur"

done
