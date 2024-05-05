import subprocess
import os

d = "mimic"
print(d)

datasets = ["mimic", "chexpert", "padchest", "nih"]
train_nums = [10, 50, 100, 250, 500, 1000]

# New script 
# Train model on MIMIC
program = f'''
        python run_cutmix_mixup.py --model_name {d} --output_dir "/data/healthy-ml/scratch/qixuanj/generative_validation/baseline_results_new3" --train_model --epochs 10 --checkpoint_epochs 2 --seed 0
        '''
subprocess.call(program, shell=True, executable='/bin/bash',)