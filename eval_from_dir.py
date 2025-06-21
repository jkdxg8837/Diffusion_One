from torchmetrics.functional.multimodal import clip_score
from functools import partial
import argparse
import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from diffusers.utils import load_image
from torchmetrics.image.fid import FrechetInceptionDistance
# Load the images and prompts
unique_token = "sks"
class_token = "dog"
prompt_list = [
'a {0} {1} in the jungle'.format(unique_token, class_token),
'a {0} {1} in the snow'.format(unique_token, class_token),
'a {0} {1} on the beach'.format(unique_token, class_token),
'a {0} {1} on a cobblestone street'.format(unique_token, class_token),
'a {0} {1} on top of pink fabric'.format(unique_token, class_token),
'a {0} {1} on top of a wooden floor'.format(unique_token, class_token),
'a {0} {1} with a city in the background'.format(unique_token, class_token),
'a {0} {1} with a mountain in the background'.format(unique_token, class_token),
'a {0} {1} with a blue house in the background'.format(unique_token, class_token),
'a {0} {1} on top of a purple rug in a forest'.format(unique_token, class_token),
'a {0} {1} wearing a red hat'.format(unique_token, class_token),
'a {0} {1} wearing a santa hat'.format(unique_token, class_token),
'a {0} {1} wearing a rainbow scarf'.format(unique_token, class_token),
'a {0} {1} wearing a black top hat and a monocle'.format(unique_token, class_token),
'a {0} {1} in a chef outfit'.format(unique_token, class_token),
'a {0} {1} in a firefighter outfit'.format(unique_token, class_token),
'a {0} {1} in a police outfit'.format(unique_token, class_token),
'a {0} {1} wearing pink glasses'.format(unique_token, class_token),
'a {0} {1} wearing a yellow shirt'.format(unique_token, class_token),
'a {0} {1} in a purple wizard outfit'.format(unique_token, class_token),
'a red {0} {1}'.format(unique_token, class_token),
'a purple {0} {1}'.format(unique_token, class_token),
'a shiny {0} {1}'.format(unique_token, class_token),
'a wet {0} {1}'.format(unique_token, class_token),
'a cube shaped {0} {1}'.format(unique_token, class_token)
]







class compute_metrics():
    def __init__(self, img_path):
        self.img_path = img_path
        self.prompt_list = prompt_list
        self.clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
        self.image_path = img_path
        self.ref_images = []
        for img_name in os.listdir("/dcs/pg24/u5649209/data/workspace/diffusers/slurm/dog"):
            img_path = os.path.join("/dcs/pg24/u5649209/data/workspace/diffusers/slurm/dog", img_name)
            if img_path.endswith('.jpeg') or img_path.endswith('.png'):
                img = load_image(img_path).resize((512, 512))
                self.ref_images.append(img)
        # repete self.ref_image 20 times to expand the 5 image list to 100 images list
        # repeat self.ref_images 20 times to expand the 5 image list to 100 images list
        print("*****")
        print(self.ref_images)
        self.ref_images = self.ref_images * 20
        self.ref_images = self.ref_images[:100]
        
    def calculate_clip_score(self, images, prompts):
        # images_int = (images * 255).astype("uint8")
        print(torch.from_numpy(images).permute(0, 3, 1, 2).shape)
        print(len(prompts))
        clip_score1 = self.clip_score_fn(torch.from_numpy(images).permute(0, 3, 1, 2), prompts).detach()
        return round(float(clip_score1), 4)
    def compute_clip(self):
        prompt_idx = 0
        
        clip_score_list = []
        img_list_I = []
        for prompt in self.prompt_list:
            print(prompt)
            img_list = []
            # 25*4 images in one array
            
            img_dir = os.path.join(self.image_path, str(prompt_idx))
            if not os.path.exists(img_dir):
                print(f"Directory {img_dir} does not exist.")
                continue
            # load images from this directory
            single_prompt_list = []
            for img_name in os.listdir(img_dir):
                img_path = os.path.join(img_dir, img_name)
                if img_path.endswith('.jpg') or img_path.endswith('.png'):
                    img = load_image(img_path).resize((512, 512))
                    img_list.append(img)
                    img_list_I.append(img)
                single_prompt_list.append(prompt)
            clip_score_list.append(self.calculate_clip_score(np.array(img_list), single_prompt_list))
        
        print(clip_score_list)
        print(f"CLIP scoreT: {np.array(clip_score_list).mean()}")
        print(np.array(img_list_I).shape)
        print(f"CLIP scoreI: {self.calculate_clip_score(np.array(img_list_I), torch.from_numpy(np.array(self.ref_images)))}")
        return clip_score_list
    def compute_DINO(self):
        pass
cm_matrics = compute_metrics(img_path="/dcs/pg24/u5649209/data/workspace/diffusers/slurm/trained-sd3-lora-fixed_0.5/checkpoint-1/output")
cm_matrics.compute_clip()