import os
from datasets import load_dataset
from PIL import Image, ImageOps
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Preprocessing script to prepare images to be trained on
# images are too big to load into memory and preprocess in the script

# Configurations
TARGET_IMAGE_SIZE = (320, 320)
OUTPUT_DIR = "./cnn_processed_dataset"
HF_DATASET_ID = "SABR22/Canadian-streetview-cities"
K_FOLD_CSV = "image_folds.csv"
N_FOLDS = 5

def resize_and_pad(img, target_size):
    # Reisze images to target size and fill space to form a square

    # By default keeps aspect ration, LANCZOS seems best for downsampling
    # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#filters
    img.thumbnail(target_size, Image.Resampling.LANCZOS)

    # Create black background
    new_img = Image.new("RGB", target_size, (0, 0, 0))

    # Create new square image with black background centered
    left = (target_size[0] - img.size[0]) // 2
    top = (target_size[1] - img.size[1]) // 2
    new_img.paste(img, (left, top))
    
    return new_img

def process_dataset():
    # Download and process images for training
    city_ds = load_dataset(HF_DATASET_ID, split="train")

    print(f"Total images: {len(city_ds)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    city_names = city_ds.features["label"].names

    for city_name in city_names:
        os.makedirs(os.path.join(OUTPUT_DIR, city_name), exist_ok=True)

    # Process images
    for i, object in tqdm(enumerate(city_ds), total=len(city_ds)):
        try:
            image = object["image"]
            city_id = object["label"]
            city_name = city_names[city_id]

            processed_img = resize_and_pad(image, TARGET_IMAGE_SIZE)

            save_path = os.path.join(OUTPUT_DIR, city_name, f"{i}.jpg")
            processed_img.save(save_path, "JPEG", quality=90)
        except Exception as e:
            print(f"Skipping image {i} due to error: {e}")

def create_folds():
    filepaths = []
    labels = []

    # Collecting from what was downloaded in casse city labels changed in future, despite we know them now
    cities = sorted([city_dir for city_dir in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, city_dir))])

    for city_name in cities:
        city_path = os.path.join(OUTPUT_DIR, city_name)
        files = os.listdir(city_path)
        for filename in files:
            if filename.lower().endswith(("jpg")):
                filepaths.append(os.path.join(city_name, filename))
                labels.append(city_name)

    # Create pandas frame with our data
    city_df = pd.DataFrame({
        "filepath": filepaths,
        "label": labels
    })

    print(f"Total files: {len(city_df)}")

    # Create predefined and reproducible folds and save
    strat_k_fold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    for fold_id, (train_id, val_id) in enumerate(strat_k_fold.split(city_df, city_df["label"])):
        city_df.loc[val_id, 'fold'] = fold_id

    city_df.to_csv(os.path.join(OUTPUT_DIR, K_FOLD_CSV), index=False)

# Processing calls
# process_dataset()

create_folds()
