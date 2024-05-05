import subprocess
import os

d = "mimic"
print(d)

datasets = ["mimic", "chexpert", "padchest", "nih"]
train_nums = [10, 50, 100, 250, 500, 1000]

# New script 
# Train model on MIMIC
program = f'''
        python run_cutmix_mixup.py --model_name {d} --output_dir "/data/healthy-ml/scratch/qixuanj/generative_validation/mixup_results_new3" --use_mixup --train_model --epochs 10 --checkpoint_epochs 2 --seed 0
        '''
subprocess.call(program, shell=True, executable='/bin/bash',)

# Transfer match


# Transfer balanced 


# Evaluate



## OLD script running ---------------------------------------------------------------------
# Train models 
# for t in datasets: 
#     if d == t: 
#         continue
#     for train_num in train_nums: 
#         program = f'''
#         python run_cutmix_mixup.py --model_name {d} --dataset_name {t} --output_dir "/data/healthy-ml/scratch/qixuanj/generative_validation/mixup_results_new2" --use_mixup --train_model --freeze_encoder --epochs 5 --train_num {train_num} --seed 0
#         '''
#         subprocess.call(program, shell=True, executable='/bin/bash',)

# # Train models balanced
# for t in datasets: 
#     if d == t: 
#         continue
#     for train_num in train_nums: 
#         program = f'''
#         python run_cutmix_mixup.py --model_name {d} --dataset_name {t} --output_dir "/data/healthy-ml/scratch/qixuanj/generative_validation/mixup_results_new2" --use_mixup --train_model --freeze_encoder --epochs 5 --train_num {train_num} --seed 0 --class_balanced
#         '''
#         subprocess.call(program, shell=True, executable='/bin/bash',)

# # Evaluate 
# for t in datasets: 
#     if d == t: 
#         continue
#     for train_num in train_nums:
#         program = f'''
#         python run_cutmix_mixup.py --model_name {d} --dataset_name {t} --output_dir "/data/healthy-ml/scratch/qixuanj/generative_validation/mixup_results_new2" --eval_model --model_path "/data/healthy-ml/scratch/qixuanj/generative_validation/mixup_results_new2/models/{train_num}/0/{d}_{t}_match_epoch5_model.pt" --epochs 5 --train_num {train_num} --seed 0
#         '''
#         subprocess.call(program, shell=True, executable='/bin/bash',)
        
#         program = f'''
#         python run_cutmix_mixup.py --model_name {d} --dataset_name {t} --output_dir "/data/healthy-ml/scratch/qixuanj/generative_validation/mixup_results_new2" --eval_model --model_path "/data/healthy-ml/scratch/qixuanj/generative_validation/mixup_results_new2/models/{train_num}/0/{d}_{t}_balanced_epoch5_model.pt" --epochs 5 --train_num {train_num} --seed 0 --class_balanced
#         '''
#         subprocess.call(program, shell=True, executable='/bin/bash',)