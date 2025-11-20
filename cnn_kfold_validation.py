import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision.transforms import v2
from PIL import Image
import timm
from tqdm import tqdm

from training_reporting import format_data, plot_loss, confusion_matrix_gen, accuracy_f1_metrics

# General Configurations
DATA_ROOT = "./cnn_processed_dataset"
CSV_FILE = "./cnn_processed_dataset/image_folds.csv"
RESULTS_DIR = "./cnn_validation_results"
NUM_WORKERS = 4
IMG_SIZE = (320, 320)

# Hyperparameter Sets
HYPER_PARAM_SETS = {
    "set_1": {
        "lr_classifier": 1e-3,
        "lr_finetune": 1e-4,
        "weight_decay": 1e-4,
        "classifier_epochs": 1,
        "finetune_epochs": 6,
        "batch_size": 32,
        "accumulation_steps": 1
    },
    "set_2": {
        "lr_classifier": 5e-4,
        "lr_finetune": 5e-5,
        "weight_decay": 0.01,
        "classifier_epochs": 1,
        "finetune_epochs": 6,
        "batch_size": 32,
        "accumulation_steps": 1
    },
    "set_3": {
        "lr_classifier": 1e-3,
        "lr_finetune": 3e-4,
        "weight_decay": 1e-5,
        "classifier_epochs": 0,
        "finetune_epochs": 8,
        "batch_size": 32,
        "accumulation_steps": 1
    }
}

# Define Dataset
class CitiesDataset(Dataset):
    def __init__(self, city_df, data_dir, transform=None):
        self.city_df = city_df
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted(city_df["label"].unique())
        self.label_map = {label: i for i, label in enumerate(self.classes)}

    def __len__(self):
        return len(self.city_df)

    def __getitem__(self, idx):
        row = self.city_df.iloc[idx]
        img_path = os.path.join(self.data_dir, row["filepath"])
        image = Image.open(img_path).convert("RGB")
        label = self.label_map[row["label"]]

        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

# Image transforms 
def get_transforms(is_train=True):
    # Default transforms: https://docs.pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_getting_started.html#i-just-want-to-do-image-classification
    # https://docs.pytorch.org/vision/stable/transforms.html#
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if is_train:
        return v2.Compose([
            v2.ToImage()
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(0.1, 0.1),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ])
    return v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=mean, std=std),])

# Return the timm model format, freeze for transfer learning
def get_model(model_name, num_classes, freeze_backbone=False):
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

    # https://stackoverflow.com/questions/73531958/freezing-certain-layers-in-neural-networks-using-pytorch-image-models
    if freeze_backbone:
        for param in model.parameters():
            para.requires_grad = False
        for param in model.get_classifier().parameters():
            param.requires_grad = True

    return model

def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True

    return model

# Training loop for epoch
# Some helpful pieces: https://medium.com/biased-algorithms/cross-validation-in-pytorch-2f9f9fa9ab16
# Mixed precision docs: https://docs.pytorch.org/docs/stable/amp.html
# Helpful tutorials: https://docs.pytorch.org/docs/stable/notes/amp_examples.html
def train_one_epoch(model, data_loader, optimizer, loss_fn, device, scaler, accum_steps):
    model.train()
    current_loss = 0.0

    for i, (images, labels) in enumerate(tqdm(data_loader)):
        images, labels = images.to(device), labels.to(device)
        # Autocast to improve memory efficiency on our resource constrained VM, also needs the scaler
        with autocast():
            outputs = model(images)
            # Accumulation steps helps for potential bigger model like EfficientNet-B6 which we looked at
            loss = loss_fn(outputs, labels) / accum_steps
        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        current_loss += loss.item() * accum_steps

    return current_loss / len(data_loader)

# Validation loop, similar to training
def validate(model, data_loader, loss_fn, device):
    model.eval()
    current_loss = 0.0

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = loss_fn(outputs, labels)
            current_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    return current_loss / len(data_loader), all_predictions, all_targets

# Main control function
def main():
    # Get configurations on which model to run
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["tf_efficientnetv2_s", "convnext_tiny", "efficientnet_b6"])
    parser.add_argument("--hyperparam", type=str, required=True, choices=["set_1", "set_2", "set_3"])
    args = parser.parse_args()

    model_name = args.model
    hp_set = HYPER_PARAM_SETS[args.hyperparam]

    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # May have to make changes for EfficientNet-B6 likely to OOM
    batch_size = hp_set["batch_size"]
    accumulation_steps = hp_set["accumulation_steps"]

    data_splits_df = pd.read_csv(CSV_FILE)
    fold_results = []

    city_names = sorted([name for name in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, name))])

    print(f"Performing 5-Fold Cross-Validation for {model_name} on {hp_set}.")

    # Driver 5-fold loop
    for fold in range(5):
        print(f"\nFold {fold}:")
        train_split_df = data_splits_df[data_splits_df["fold"] != fold]
        validation_split_df = data_splits_df[data_splits_df["fold"] == fold]

        train_loader = DataLoader(
            CitiesDataset(train_split_df, DATA_ROOT, transform=get_transforms(is_train=True)),
            batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
        )
        validation_loader = DataLoader(
            CitiesDataset(validation_split_df, DATA_ROOT, transform=get_transforms(is_train=False)),
            batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
        )

        # Create the model for this fold
        model = get_model(model_name, len(city_names), freeze_backbone=True).to(device)
        loss_func = nn.CrossEntropyLoss()
        scaler = GradScaler()

        # Results for this fold
        fold_result = {"training_loss": [], "validation_loss": [], "final_predictions": [], "final_targets": []}

        # Training of classifier (if scheduled)
        # AdamW suggested for better generalization which seems applicable to our diverse dataset and task
        if hp_set["classifier_epochs"] > 0:
            optimizer = optim.AdamW(model.parameters(), lr=hp_set["lr_classifier"], weight_decay=hp_set["weight_decay"])
            for epoch in range(hp_set["classifier_epochs"]):
                training_loss = train_one_epoch(model, train_loader, optimizer, loss_func, device, scaler, accumulation_steps)
                validation_loss = validate(model, validation_loader, loss_func, device)
                
                fold_result["training_loss"].append(training_loss)
                fold_result["validation_loss"].append(validation_loss)

        if hp_set["finetune_epochs"] > 0:
            model = unfreeze_model(model)
            # Splitting learing rates so new classifier keeps learning at faster rate while making smaller changes to whole network
            optimizer = optim.AdamW([
                {"params": model.get_classifier().parameters(), "lr": hp["lr_classifier"]},
                {"params": model.features.parameters() if hasattr(model, "features") else model.parameters(), "lr": hp["lr_finetune"]}
            ], weight_decay=hp_set["weight_decay"])

            best_validation_accuracy = 0

            for epoch in range(hp_set["finetune_epochs"]):
                # NOTE Complete similar to previous loop but save results of best performing model

if __name__ == "__main__":
    main()