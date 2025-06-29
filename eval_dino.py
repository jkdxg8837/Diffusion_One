import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import sqrtm

# 检查是否有GPU可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 关键修改部分：加载 DINO-S/16 模型 ---
model_name = "dinov2_vits14" # 对应 DINO-S/16 (Vision Transformer Small, 14x14 patch size)
model = torch.hub.load('facebookresearch/dinov2', model_name).to(device)
model.eval() # 将模型设置为评估模式

print(f"Loaded DINOv2 model: {model_name}")

# DINO模型的标准预处理步骤 (保持不变)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
])
print("Defined image transformations.")

# 提取图像特征的函数 (保持不变)
def extract_dino_features(image_paths, model, transform, device):
    features_list = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feature = model(img_tensor).squeeze(0).cpu().numpy()
                features_list.append(feature)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    return np.array(features_list)

print("Defined feature extraction function.")

# 计算DINO相似度（用于实例保真度） (保持不变)
def calculate_dino_similarity(generated_features, reference_features):
    similarities = []
    for gen_feat in generated_features:
        sims_with_refs = cosine_similarity(gen_feat.reshape(1, -1), reference_features)
        similarities.append(np.max(sims_with_refs))
    return np.mean(similarities)

print("Defined DINO similarity calculation function.")

# 计算Fréchet DINO Distance (FD-DINO) (保持不变)
def calculate_fid(features1, features2):
    mu1, sigma1 = np.mean(features1, axis=0), np.cov(features1, rowvar=False)
    mu2, sigma2 = np.mean(features2, axis=0), np.cov(features2, rowvar=False)

    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

print("Defined FID (FD-DINO) calculation function.")

# --- 示例：运行评估流程 (保持不变，但请替换你的图片路径) ---

reference_image_dir = "path/to/your/reference_images"
generated_image_dir = "path/to/your/generated_images"

# 获取所有图片路径
reference_image_paths = [os.path.join(reference_image_dir, f) for f in os.listdir(reference_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
generated_image_paths = [os.path.join(generated_image_dir, f) for f in os.listdir(generated_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"Found {len(reference_image_paths)} reference images.")
print(f"Found {len(generated_image_paths)} generated images.")

# 提取特征
print("Extracting DINO features from reference images...")
reference_features = extract_dino_features(reference_image_paths, model, transform, device)
print(f"Reference features shape: {reference_features.shape}")

print("Extracting DINO features from generated images...")
generated_features = extract_dino_features(generated_image_paths, model, transform, device)
print(f"Generated features shape: {generated_features.shape}")

# 计算DINO相似度 (实例保真度)
if len(generated_features) > 0 and len(reference_features) > 0:
    dino_similarity_score = calculate_dino_similarity(generated_features, reference_features)
    print(f"\n--- Evaluation Results ---")
    print(f"Average DINO Similarity (higher is better): {dino_similarity_score:.4f}")
else:
    print("\nNot enough images to calculate DINO Similarity. Make sure image paths are correct.")

# 计算Fréchet DINO Distance (FD-DINO)
# 注意：对于少量参考图片 (例如5张)，FD-DINO可能不准确。
if len(generated_features) >= reference_features.shape[1] and len(reference_features) >= reference_features.shape[1]:
    try:
        fd_dino_score = calculate_fid(reference_features, generated_features)
        print(f"Fréchet DINO Distance (lower is better): {fd_dino_score:.4f}")
    except Exception as e:
        print(f"Could not calculate FD-DINO (possibly due to too few samples for covariance matrix estimation): {e}")
else:
    print("\nNot enough images to reliably calculate FD-DINO. Consider using more samples.")