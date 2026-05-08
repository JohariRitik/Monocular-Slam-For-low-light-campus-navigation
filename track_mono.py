import os
import numpy as np
from PIL import Image
import rerun as rr
import rerun.blueprint as rrb
import cuvslam

dataset_path = "/datasets/myphone"

# 1. Setup Rerun live streaming
rr.init('phone_mono', strict=True, spawn=False)
rr.connect_tcp()

# 2. Setup Camera Calibration
camera = cuvslam.Camera()
camera.size = (3840, 2160)
camera.focal = [2560.0,2560.0] #[640.0, 640.0]        # Calculated from the LYT-700C & 24mm equivalent
camera.principal = [1920.0, 1080.0]    # Exact center of your 540x960 frame

# 3. Initialize the tracker configuration
cfg = cuvslam.Tracker.OdometryConfig(
    async_sba=True,
    enable_final_landmarks_export=True,
    odometry_mode=cuvslam.Tracker.OdometryMode.Mono 
)

# MIRNet already denoised it! Don't let the engine blur it again.
cfg.use_denoising = False 

# --- THE FIX ---
# Removed "Tracker." from the MappingMode enum to fix the AttributeError
cfg.use_motion_model = True
cfg.max_frame_delta_s = 0.5
#cfg.max_features = 3000
#cfg.max_epipolar_error = 1.0
tracker = cuvslam.Tracker(cuvslam.Rig([camera]), cfg)

# 4. Load Timestamps
timestamps = [
    int(10 ** 9 * float(sec_str))
    for sec_str in open(os.path.join(dataset_path, 'daytime_images1_times.txt')).readlines()
]
import glob

# 4. Generate Perfect 10 FPS Timestamps
# Count exactly how many images are in your enhanced folder
image_files = sorted(glob.glob(os.path.join(dataset_path, 'daytime_images_1', '*.png')))
total_images = len(image_files)

# Generate a perfect 0.1 second (10 FPS) step in nanoseconds
#timestamps = [int(frame_idx * 0.1 * 1e9) for frame_idx in range(total_images)]

#print(f"Found {total_images} images. Generated perfect 10 FPS timestamps.")
# 5. Track the Video!
trajectory = []
print("Starting Monocular SLAM...")

for frame in range(60, total_images):
    #img_path = os.path.join(dataset_path, 'image_0', f'{frame:06d}.png')
    img_path = image_files[frame]
    if not os.path.exists(img_path):
        print(f"\n[STOP] Reached the end of available images at frame {frame}.")
        print("Stopping SLAM tracking cleanly.")
        break  # Exit the loop instead of crashing
    
    # cuVSLAM tracks best with grayscale images
    gray_image = np.asarray(Image.open(img_path).convert('L'))

    # Send it to the engine
    odom_pose_estimate, _ = tracker.track(timestamps[frame], [gray_image])

    if odom_pose_estimate.world_from_rig is None:
        print(f"Warning: Failed to track frame {frame} (Usually due to motion blur or fast panning)")
        continue

    odom_pose = odom_pose_estimate.world_from_rig.pose
    trajectory.append(odom_pose.translation)

    # Stream to Rerun
    rr.set_time_sequence('frame', frame)
    rr.log('trajectory', rr.LineStrips3D(trajectory))
    rr.log('camera/pose', rr.Transform3D(
        translation=odom_pose.translation,
        quaternion=odom_pose.rotation
    ))
    
    # Stream the original color image for the visualizer
    color_img = np.asarray(Image.open(img_path))
    rr.log('camera/image', rr.Image(color_img).compress(jpeg_quality=80))

print("Finished tracking!")
