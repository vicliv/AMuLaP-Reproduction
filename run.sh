#!/bin/bash
##SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-54
#SBATCH --output=output/experiment-%A.%a.out
#SBATCH --error=output/error/experiment-%A.%a.out

echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"

# make sure to change the environment
module load python/3.7
source ~/.virtualenvs/amulap/bin/activate

seed_arr=(13 21 42 87 100)
#seed_arr=(42)
seed_len=${#seed_arr[@]}

#k_arr=(1 2 4 8 16 32 64 128 256 512 1024)
k_arr=(1 2 4 8 16)
k_len=${#k_arr[@]}

path=data/k-shot/CoLA/16-

for ((idx_1=0; idx_1<$seed_len; idx_1++))
do
    for ((idx_2=0; idx_2<$k_len; idx_2++))
    do
        task_id=`expr $idx_1 \* $k_len + $idx_2`
        data=$path${seed_arr[$idx_1]}

        #check for the correct task id
        if [ $task_id == $SLURM_ARRAY_TASK_ID ]
        then
            srun python -u run_prompt.py --model_name_or_path roberta-large --task_name cola --data_dir $data   --output_dir outputs_large     --shot_num 16    --max_train_steps 1000     --num_warmup_steps 0     --eval_steps 100     --learning_rate 5e-5     --per_device_train_batch_size 4     --per_device_eval_batch_size 4     --top_k ${k_arr[$idx_2]}     --max_seq_len 128 --template *cls**sent_0*_This_is*mask*.*sep+* --seed ${seed_arr[$idx_1]} --exp_id $SLURM_ARRAY_TASK_ID --dedup AMuLaP
        fi
    done
done