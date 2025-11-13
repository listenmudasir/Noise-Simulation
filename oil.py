import cv2
import os
import numpy as np
import math
from pathlib import Path
from tqdm import tqdm  # ✅ Progress bar

# === (1) Radial Gradient（已存在）===
def add_ir_radial_f(im, inner_radius_factor=0.25, outer_radius_factor=0.7,
                    inner_brightness=0.8, outer_brightness=0.05):
    mask = np.zeros(im.shape[:2], dtype=np.float32)
    center = (mask.shape[1] // 2, mask.shape[0] // 2)
    max_radius = math.sqrt(center[0] ** 2 + center[1] ** 2)
    inner_radius = inner_radius_factor * max_radius
    outer_radius = outer_radius_factor * max_radius

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            distance = math.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
            if distance <= inner_radius:
                mask[y, x] = inner_brightness
            elif distance <= outer_radius:
                alpha = (distance - inner_radius) / (outer_radius - inner_radius)
                mask[y, x] = (inner_brightness * (1 - alpha)
                              + outer_brightness * alpha)
            else:
                mask[y, x] = outer_brightness

    mask_3channel = cv2.merge([mask]*3)
    im_float = im.astype(np.float32) / 255.0
    result = im_float * mask_3channel
    result = (result * 255).astype(np.uint8)
    return result

# === (2) 加雜訊（已存在）===
def add_grainy_f(im, noise_strength=0.45):
    noise = np.random.normal(0.5, 0.2, im.shape).astype(np.float32)
    im_float = im.astype(np.float32) / 255.0
    grainy_image = cv2.addWeighted(im_float, 1.0, noise * noise_strength,
                                   noise_strength, 0)
    return (grainy_image * 255).astype(np.uint8)

# === (3) 降對比（已存在）===
def add_contrast_f(im, contrast_strength=0.7):
    im_float = im.astype(np.float32) / 255.0
    gray = np.full(im.shape, 0.5, dtype=np.float32)
    low_contrast_image = cv2.addWeighted(im_float, contrast_strength, gray,
                                         1 - contrast_strength, 0)
    return (low_contrast_image * 255).astype(np.uint8)

# === (4) 轉灰階（已存在）===
def convert_to_grayscale(im):
    gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

# === (5) 油畫濾鏡（需 opencv-contrib）===
def apply_oil_painting(im, size=6, dyn_ratio=1):
    return cv2.xphoto.oilPainting(im, size, dyn_ratio)

# === (6) 主處理流程 ===
def process_images(input_dir, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    frame_counter = 0

    img_files = [img for img in Path(input_dir).rglob("*") if img.suffix.lower() in [".jpg", ".jpeg", ".png"]]

    for img_file in tqdm(img_files, desc="Processing Images"):
        im = cv2.imread(str(img_file))
        if im is None:
            print(f"Error: Could not open image file {img_file}")
            continue

        # 1) 灰階
        gray_frame = convert_to_grayscale(im)

        # 2) 模糊
        processed_frame = cv2.blur(gray_frame, (7, 7))

        # 3) 壓暗
        processed_frame = cv2.convertScaleAbs(processed_frame, alpha=0.5, beta=-10)

        # 4) 降對比
        processed_frame = add_contrast_f(processed_frame, contrast_strength=0.4)

        # 5) 加入 Radial Gradient
        processed_frame = add_ir_radial_f(
            processed_frame,
            inner_radius_factor=0.25,
            outer_radius_factor=0.7,
            inner_brightness=0.7,
            outer_brightness=0.05
        )

        # 6) 加入粒狀雜訊
        processed_frame = add_grainy_f(processed_frame, noise_strength=0.4)

        # 7) 油畫濾鏡
        processed_frame = apply_oil_painting(processed_frame, size=7, dyn_ratio=1)

        # 8) 存檔
        processed_filename = os.path.join(output_dir, img_file.name)
        cv2.imwrite(processed_filename, processed_frame)
        frame_counter += 1

    print(f"Processing complete. Saved {frame_counter} images in {output_dir}")

if __name__ == "__main__":
    input_dir = 'Maindir/train'
    output_dir = 'Maindir/train_noisy'
    process_images(input_dir, output_dir)
