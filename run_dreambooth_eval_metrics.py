import torch
import os
import cv2
from PIL import Image
from tqdm import tqdm 
import numpy as np
import pandas as pd 
from glob import glob
import subprocess
import gc 
import shutil

from diffusers import StableDiffusionPipeline
from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel

checkpoint_prefix = "runwayml_imagenet_"
checkpoint_names = [
                    'class0_tench',
                    # 'class0_fisherman',
                    #  'class0_fishing-net',
                     # 'class0_grass',
                     # 'class0_lake-surface',
                     # 'class2_ocean',
                     # 'class2_shark',
                     # 'class324_flower',
                     # 'class325_flower',
                     # 'class384_leaves',
                     # 'class384_tree-trunk',
                     # 'class434_baby',
                     # 'class434_bath-towel',
                     # 'class80_black-grouse',
                     # 'class80_prairie',
                     # 'class94_bird-feeder',
                     # 'class94_branch',
                     # 'class94_flower',
                     # 'class94_hummingbird', 
                     # 'class94_bird-feeder', 
                     # 'class325_sulphur-butterfly', 
                     # 'class324_cabbage-butterfly', 
                     # 'class384_indri', 
                   ]
checkpoint_nums = ["500", "1000", "1500", "2000", "2500"]

# Code from https://stackoverflow.com/questions/50331463/convert-rgba-to-rgb-in-python 
def rgba2rgb( rgba, background=(0,0,0) ):
    row, col, ch = rgba.shape
    if ch == 3:
        return rgba
    assert ch == 4, 'RGBA image has 4 channels.'
    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background
    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B
    return np.asarray( rgb, dtype='uint8' )



for checkpoint_name in checkpoint_names: 
    for checkpoint_num in checkpoint_nums: 
        print(f"{checkpoint_name} with checkpoint {checkpoint_num}")
        file_prefix = f"/data/scratch/qixuanj/imagenet_runwayml_dreambooth_ckpts/{checkpoint_prefix}{checkpoint_name}/checkpoint-{checkpoint_num}"

        if not os.path.exists(file_prefix): 
            print("Model checkpoint does not exist!") 
            print(file_prefix)
            continue
        
        model_id = "runwayml/stable-diffusion-v1-5"
        unet = UNet2DConditionModel.from_pretrained(f"{file_prefix}/unet")
        text_encoder = CLIPTextModel.from_pretrained(f"{file_prefix}/text_encoder")
        
        pipe = DiffusionPipeline.from_pretrained(model_id, unet=unet, text_encoder=text_encoder, dtype=torch.float16)
        pipe.to("cuda")

        num_images = 50
        # Keep running model to remove black images from NSFW trigger 
        images = pipe(prompt=f"a photo of <{checkpoint_name}>", 
                      negative_prompt="",
                      strength=0.9, guidance_scale=7.5, num_inference_steps=50, 
                      num_images_per_prompt=num_images).images
        
        max_tries = 15
        prev_num_empty = 0
        while max_tries > 0: 
            num_empty = 0
            for img in images: 
                if np.array(img).mean() == 0: 
                    num_empty += 1
                    images.remove(img)
            # Also break out of loop if keep generating same number of black images 
            if num_empty == prev_num_empty: 
                break 
            else: 
                prev_num_empty = num_empty
        
            # No missing images to generate 
            if num_empty == 0: 
                break
            images += pipe(prompt=f"a photo of <{checkpoint_name}>", 
                      negative_prompt="",
                      strength=0.9, guidance_scale=7.5, num_inference_steps=50, 
                      num_images_per_prompt=num_empty).images
            max_tries -= 1
        for img in images: 
            if np.array(img).mean() == 0: 
                num_empty += 1
                images.remove(img)
        print(f"Max tries {max_tries} left out of 15; Total {len(images)} images")

        output_dir = f"imagenet_runwayml_dreambooth_imgs/{checkpoint_prefix}{checkpoint_name}/checkpoint-{checkpoint_num}" 
        if not os.path.exists(output_dir): 
            os.makedirs(output_dir) 
        seg_dir = output_dir + "-masked/"

        for i, image in enumerate(images):
            image.save(output_dir + f"/image{i}.png")

        output_dir = os.getcwd() + "/" + output_dir
        seg_dir = os.getcwd() + "/" + seg_dir 

        program = f'''source ~/.bashrc
                    conda activate spurious_imagenet
                    cd U2_Net
                    python u2net_test_copy.py --img_dir "{output_dir}" --prediction_dir "{seg_dir}"
                    '''
        subprocess.call(program, shell=True, executable='/bin/bash',)
        print("Background removal completed!")

        preprocess_dir = output_dir + "-preprocessed"
        if not os.path.exists(preprocess_dir): 
            os.makedirs(preprocess_dir) 

        img_paths = glob(output_dir + "/*.png", recursive = True)

        mask_threshold = 0.8

        for img_path in img_paths: 
            i = int(img_path.split(".png")[0].split("image")[-1])
            image = cv2.imread(img_path)
            mask_raw = cv2.imread(img_path.replace(f"checkpoint-{checkpoint_num}", 
                                                   f"checkpoint-{checkpoint_num}-masked"))
            mask_raw = mask_raw / 255 
            mask_raw[mask_raw > mask_threshold] = 1
            mask_raw[mask_raw <= mask_threshold] = 0
        
            shape = mask_raw.shape
            a_layer_init = np.ones(shape = (shape[0],shape[1],1))
            mul_layer = np.expand_dims(mask_raw[:,:,0],axis=2)
            a_layer = mul_layer*a_layer_init
            rgba_out = np.append(mask_raw,a_layer,axis=2)
        
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image / 255
            a_layer = np.ones(shape = (shape[1],shape[0],1))
            rgba_inp = np.append(image,a_layer,axis=2)
        
            rem_back = 255*(rgba_inp*rgba_out)
            rem_back = rgba2rgb(rem_back)
        
            rem_back_scaled = Image.fromarray(rem_back.astype('uint8'), 'RGB')
            rem_back_scaled.save(preprocess_dir + f"/images{i}.png")

        original_dir_renamed = "/data/healthy-ml/scratch/qixuanj/generative_validation/dgm_eval_source_images"
        shutil.rmtree(original_dir_renamed)
        if not os.path.exists(original_dir_renamed): 
            os.makedirs(original_dir_renamed)
        # Need to convert from jpeg to png 
        tmp_dir = "/data/healthy-ml/scratch/qixuanj/generative_validation/spurious_imagenet/dataset/sam_masks_grouping_histogram_fixed/" 
        tmp1 = checkpoint_name.split("_")[0]
        tmp2 = checkpoint_name.split("_")[1]
        original_dir = glob(tmp_dir + f"imagenet_{tmp1}/{tmp2}*")[0]
        img_paths = glob(original_dir + "/*.JPEG", recursive = True)
        for i, img_path in enumerate(img_paths): 
            img = Image.open(img_path) 
            img.save(original_dir_renamed + f"/{i}.png")

        metrics_dir = output_dir + "-metrics/"
        if not os.path.exists(metrics_dir): 
            os.makedirs(metrics_dir)

        # Classic inception-v3 model 
        program = f"""
                    source ~/.bashrc
                    conda activate dgm-eval
                    cd dgm-eval
                    python -m dgm_eval "{original_dir_renamed}" "{preprocess_dir}" \
                    				--model inception \
                                    --device cuda \
                                    --batch_size 5 \
                                    --output_dir "{metrics_dir}" \
                    				--metrics fd kd prdc authpct 
                    """
        subprocess.call(program, shell=True, executable='/bin/bash',)

        # DinoV2 model 
        program = f"""
                    source ~/.bashrc
                    conda activate dgm-eval
                    cd dgm-eval
                    python -m dgm_eval "{original_dir_renamed}" "{preprocess_dir}" \
                    				--model dinov2 \
                                    --device cuda \
                                    --batch_size 5 \
                                    --output_dir "{metrics_dir}" \
                    				--metrics fd kd prdc authpct 
                    """
        subprocess.call(program, shell=True, executable='/bin/bash',)

        print("Metrics computation completed!")

        # Clean up memory 
        del unet
        del text_encoder
        del pipe
        gc.collect()