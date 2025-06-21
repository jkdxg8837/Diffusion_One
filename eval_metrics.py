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
class compute_metrics():
    def __init__(self, val_length, img_path, condition_type):
        self.val_length = val_length
        self.img_path = img_path
        self.condition_type = condition_type
        self.dataset_length = val_length
        self.dataset = load_dataset("jxie/coco_captions", split="validation")[:val_length]
        self.image_list = self.dataset["image"]
        self.prompt_list = self.dataset["caption"]
        self.real_images_smaller_array = []
        self.real_images_array = []
        for image in self.image_list:
            image = load_image(image).resize((512, 512))
            self.real_images_array.append(image)
            image = image.resize((256, 256))
            self.real_images_smaller_array.append(image)
        self.real_images_smaller_array = np.array([np.array(img) for img in self.real_images_smaller_array])
        self.real_images_array = np.array([np.array(img) for img in self.real_images_array])
        self.real_images_tensor = torch.from_numpy(self.real_images_array).permute(0, 3, 1, 2)
        self.real_images_smaller_tensor = torch.from_numpy(self.real_images_smaller_array).permute(0, 3, 1, 2)
        self.clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
        
    def calculate_clip_score(self, images, raw_images, prompts):
        # images_int = (images * 255).astype("uint8")
        clip_score1 = self.clip_score_fn(torch.from_numpy(images).permute(0, 3, 1, 2), prompts).detach()
        clip_score2 = self.clip_score_fn(torch.from_numpy(images).permute(0, 3, 1, 2), raw_images).detach()
        return round(float(clip_score1), 4), round(float(clip_score2), 4)
    def compute(self):
        generated_images = []
        generated_images_smaller = []
        for i in tqdm(range(self.dataset_length)):
            img_path = os.path.join(self.img_path, self.condition_type+"_"+str(i)+".jpg")
            image = Image.open(img_path).convert("RGB")
            image_smaller = image.resize((256, 256))
            generated_images_smaller.append(image_smaller)
            generated_images.append(image)
        generated_images = np.array([np.array(img) for img in generated_images])
        generated_images_tensor = torch.from_numpy(generated_images).permute(0, 3, 1, 2)
        sd_clip_score1, sd_clip_score2 = self.calculate_clip_score(generated_images, self.real_images_tensor, self.prompt_list)
        print(f"CLIP score1: {sd_clip_score1}")
        print(f"CLIP score2: {sd_clip_score2}")

        print(f"real_images_tensor for fid shape: {self.real_images_tensor.shape}")
        print(f"generated_images_tensor for fid shape: {generated_images_tensor.shape}")
        fid = FrechetInceptionDistance(normalize=True)
        fid.update(self.real_images_tensor, real=True)
        fid.update(generated_images_tensor, real=False)
        fid_score = fid.compute()
        print(f"FID score: {fid_score}")