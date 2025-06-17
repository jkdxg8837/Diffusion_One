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
        
    def calculate_clip_score(self, images, prompts):
        # images_int = (images * 255).astype("uint8")
        print(torch.from_numpy(images).permute(0, 3, 1, 2).shape)
        print(len(prompts))
        clip_score1 = self.clip_score_fn(torch.from_numpy(images).permute(0, 3, 1, 2), prompts).detach()
        return round(float(clip_score1), 4)
    def compute(self):
        prompt_idx = 0
        
        clip_score_list = []
        for prompt in self.prompt_list:
            print(prompt)
            img_list = []
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
                single_prompt_list.append(prompt)
            clip_score_list.append(self.calculate_clip_score(np.array(img_list), single_prompt_list))
        print(clip_score_list)
        print(f"CLIP score1: {np.array(clip_score_list).mean()}")
        return clip_score_list
cm_matrics = compute_metrics(img_path="/dcs/pg24/u5649209/data/workspace/diffusers/slurm/trained-sd3-lora-one_0.5/checkpoint-10/output")
cm_matrics.compute()