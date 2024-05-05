from glob import glob
import os
import subprocess


## NOTE: This script is for autodetecting all classes and concepts in imagenet that haven't been 
##       trained yet and use subprocess to train them sequentially. If you want to just train one 
##       model, just run the accelerate launch program locally with the appropriate parameters. 

data_dir = "/data/healthy-ml/scratch/qixuanj/generative_validation/spurious_imagenet/dataset/sam_masks_grouping_histogram_fixed"
subdirs = glob(data_dir + "/*/", recursive = True)

model_name = "runwayml"
model_path = "runwayml/stable-diffusion-v1-5"

# Change this to your local folder that you want to save the checkpoints to 
output_dir = f"/data/scratch/qixuanj/imagenet_{model_name}_dreambooth_ckpts"
if not os.path.exists(output_dir): 
    os.makedirs(output_dir)

datadirs = {}
targetdirs = {}
feature_names = {}
classes = {}

count = 0
for s in subdirs: 
    # ImageNet class 
    c = s.split("/")[-2].split("_")[1]
    # Get through subdirectories to find concept name
    subdirs2 = glob(s + "/*/", recursive = True)
    for s2 in subdirs2: 
        # Add to list of actual data directories to train script on 
        datadirs[count] = s2
        feature_name = s2.split("/")[-2].split("_")[0]
        feature_names[count] = feature_name
        targetdirs[count] = (output_dir + f"/{model_name}_imagenet_{c}_{feature_name}/")
        classes[count] = c
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
    c = classes[i]
    
    program_list.append(f'''accelerate launch /data/healthy-ml/scratch/qixuanj/generative_validation/diffusers/examples/dreambooth/train_dreambooth.py \
    --train_text_encoder \
    --instance_data_dir="{datadir}" \
    --pretrained_model_name_or_path="{model_path}" \
    --output_dir="{savedir}" \
    --instance_prompt="a photo of <{c}-{feature_name}>" \
    --resolution=512 \
    --snr_gamma=5.0 \
    --learning_rate=1.0e-06 \
    --lr_scheduler="constant" \
    --max_train_steps=2000 \
    --checkpointing_steps=500 \
    --seed=0 \
    --gradient_accumulation_steps=4 \
    --train_batch_size=2 \
    --validation_prompt="a photo of <{c}-{feature_name}>" \
    --num_validation_images=3 \
    --validation_steps=500 \
    --report_to="wandb" ''')

# Run all processes sequentially
for i, program in enumerate(program_list):
    subprocess.call(program, shell=True)
    print(f"Finished: program {i} with token <{classes[i]}-{feature_names[i]}>")