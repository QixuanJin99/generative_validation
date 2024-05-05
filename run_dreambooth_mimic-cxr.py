from glob import glob
import os
import subprocess

dataset_name = "mimic-cxr"
data_dir = "/data/healthy-ml/scratch/qixuanj/generative_validation/mimic-cxr-dreambooth-training"
subdirs = glob(data_dir + "/*/", recursive = True)

model_name = "roentgen"
model_path = "/data/healthy-ml/scratch/qixuanj/generative_validation/roentgen"

output_dir = f"/data/scratch/qixuanj/mimic-cxr_{model_name}_dreambooth_ckpts"
if not os.path.exists(output_dir): 
    os.makedirs(output_dir)

datadirs = {}
feature_names = {}
init_tokens = {} 
targetdirs = {}

count = 0
for s in subdirs:  
    feature_name = s.split("/")[-2].split("_")[0]
    init_token = s.split("/")[-2].split("_")[1]

    feature_names[count] = feature_name 
    init_tokens[count] = init_token 
    datadirs[count] = s 
    targetdirs[count] = output_dir + f"/{feature_name}_{init_token}"
    count += 1 

# If directory does not exist or is empty, rerun training 
savedirs = {}
for i, t in targetdirs.items(): 
     if not os.path.exists(t) or len(os.listdir(t)) == 0: 
         savedirs[i] = t
print(len(savedirs))
print(savedirs)

program_list = []
for i, savedir in savedirs.items(): 
    datadir = datadirs[i]
    feature_name = feature_names[i]
    
    program_list.append(f'''accelerate launch /data/healthy-ml/scratch/qixuanj/generative_validation/diffusers/examples/dreambooth/train_dreambooth.py \
    --train_text_encoder \
    --instance_data_dir="{datadir}" \
    --pretrained_model_name_or_path="{model_path}" \
    --output_dir="{savedir}" \
    --instance_prompt="a photo of <{dataset_name}-{feature_name}>" \
    --resolution=512 \
    --snr_gamma=5.0 \
    --learning_rate=1.0e-06 \
    --lr_scheduler="constant" \
    --max_train_steps=2500 \
    --checkpointing_steps=500 \
    --seed=0 \
    --gradient_accumulation_steps=4 \
    --train_batch_size=2 \
    --validation_prompt="a photo of <{dataset_name}-{feature_name}>" \
    --num_validation_images=3 \
    --validation_steps=500 \
    --report_to="wandb" ''')

# Run all processes sequentially
for i, program in enumerate(program_list):
    subprocess.call(program, shell=True)
    print(f"Finished: program {i} with token <{dataset_name}-{feature_names[i]}>")