import cv2
import os
import shutil

# --- PATHS ---
video_path = "/datasets/myphone/daytime_campus.mp4" 
output_dir = "/datasets/myphone/daytime_images_1"
times_file = "/datasets/myphone/daytime_images1_times.txt"

# --- SAFETY: CLEAR OLD FRAMES ---
print("Cleaning up old frames...")
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

# --- VIDEO SETUP ---
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open {video_path}")
    exit()

original_fps = cap.get(cv2.CAP_PROP_FPS)

# --- FRAMERATE SETTINGS ---
target_fps = 10  # 10 FPS is the sweet spot for a walking SLAM dataset
frame_skip = int(original_fps / target_fps)

print(f"Original video is {original_fps:.2f} FPS.")
print(f"Extracting at ~{target_fps} FPS (Saving every {frame_skip}th frame)...")

# --- EXTRACTION LOOP ---
with open(times_file, "w") as f:
    frame_idx = 0         # Tracks the true video frame
    saved_frame_idx = 0   # Tracks the sequential names for PyCuVSLAM (000000.png)
    
    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        
        # Only save the frame if it matches our skip interval
        if frame_idx % frame_skip == 0:
            
            # Save frame sequentially
            cv2.imwrite(os.path.join(output_dir, f"{saved_frame_idx:06d}.png"), frame)
            
            # Calculate true timestamp
            timestamp_sec = frame_idx / original_fps
            f.write(f"{timestamp_sec:.6f}\n")
            
            saved_frame_idx += 1
            
        if frame_idx % 300 == 0:
            print(f"Read {frame_idx} frames... Extracted {saved_frame_idx} so far.", end='\r')
            
        frame_idx += 1

cap.release()
print(f"\nDone! Shaved the dataset down from {frame_idx} to just {saved_frame_idx} frames.")
