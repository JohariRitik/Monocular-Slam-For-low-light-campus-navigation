import os
import shutil
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from glob import glob
from tqdm import tqdm

from networks.MIRNet_model import MIRNet
import utils # Pulling from your repo's utilities

# --- PATHS ---
INPUT_DIR = "/datasets/myphone/working2"
OUTPUT_DIR = "/datasets/myphone/enhanced_2"
WEIGHTS = "pretrained_models/model_fivek.pth" # Ensure this points to your downloaded model

# --- SAFETY: CLEAR OLD FRAMES ---
print("Cleaning up old enhanced frames...")
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)    # Deletes the whole folder and everything in it
os.makedirs(OUTPUT_DIR, exist_ok=True) # Recreates a fresh, empty folder

# --- DEVICE & MODEL SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Initializing MIRNet on {device}...")

model = MIRNet()
utils.load_checkpoint(model, WEIGHTS) # Using the clean utils loader from your notebook
model = model.to(device)
model.eval()

transform = transforms.ToTensor()

# --- GET IMAGES ---
image_files = sorted(glob(os.path.join(INPUT_DIR, "*.png")))
print(f"Found {len(image_files)} extracted frames. Starting batch enhancement...")

# --- THE BATCH LOOP ---
for img_path in tqdm(image_files, desc="Processing"):
    
    # 1. Load with PIL (from your notebook)
    img = Image.open(img_path).convert("RGB")
    width, height = img.size
    img = img.resize((width // 2, height // 2), Image.Resampling.LANCZOS)
    
    # 2. Transform to Tensor (from your notebook)
    inp = transform(img).unsqueeze(0).to(device)

    # 3. Inference
    with torch.no_grad():
        restored = model(inp)
        restored = torch.clamp(restored, 0, 1)

    # 4. Post-Process back to Image (from your notebook)
    restored = restored.squeeze(0).permute(1, 2, 0).cpu().numpy()
    restored = (restored * 255).astype(np.uint8)
    restored = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)

    # 5. Save exactly where alpha-blending expects it
    filename = os.path.basename(img_path)
    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), restored)

print("✅ Batch enhancement complete! Your frames are ready for Alpha Blending.")
