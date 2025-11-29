import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torchvision.transforms import v2
from PIL import Image
import timm
from tqdm import tqdm

from training_reporting import format_data, plot_loss, confusion_matrix_gen, accuracy_f1_metrics, plot_test

# General Configurations
# DATA_ROOT = "./cnn_processed_dataset"
# TEST_ROOT = "./cnn_testing_dataset"
# RESULTS_DIR = "./cnn_training_results"
# CSV_FILE = "./cnn_processed_dataset/image_folds.csv"
# CSV_TRAIN = "./cnn_training_dataset/test_set.csv"
DATA_ROOT = "./cnn_pre_test"
TEST_ROOT = "./cnn_pre_testing"
RESULTS_DIR = "./cnn_validation_results_pre"
CSV_FILE = "./cnn_pre_test/image_folds.csv"
CSV_TRAIN = "./cnn_pre_testing/test_set.csv"
NUM_WORKERS = 4
IMG_SIZE = (320, 320)

# Hyperparameter Set
# Chosen set 3 from convnext as its our highest performing validation
HYPER_PARAMS = {
    "convnext_tiny": {
        "set_3": {
            "lr_classifier": 5e-4, "lr_finetune": 1e-5, "weight_decay": 0.01,
            "classifier_epochs": 2, "finetune_epochs": 8,
            "batch_size": 32, "accumulation_steps": 1,
            "label_smoothing": 0.05, "use_grad_checkpoint": False
        }
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
            v2.ToImage(),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(0.1, 0.1),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ])
    return v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=mean, std=std),])

# Return the timm model format, freeze for transfer learning
def get_model(model_name, num_classes, freeze_backbone=False, grad_checkpointing=False):
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

    # Ensure B6 can still train well with small batch sizes
    if grad_checkpointing:
        model.set_grad_checkpointing(True)

    # https://stackoverflow.com/questions/73531958/freezing-certain-layers-in-neural-networks-using-pytorch-image-models
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
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
def train_one_epoch(model, data_loader, optimizer, loss_fn, device, scaler, use_tqdm):
    model.train()
    current_loss = 0.0

    # leave=False helps with screen clutter
    for i, (images, labels) in enumerate(tqdm(data_loader, leave=False, disable=not use_tqdm)):
        images, labels = images.to(device), labels.to(device)
        # Autocast to improve memory efficiency on our resource constrained VM, also needs the scaler
        with autocast(device_type=device.type):
            outputs = model(images)
            loss = loss_fn(outputs, labels)
        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        current_loss += loss.item()

    return current_loss / len(data_loader)

# Validation loop, similar to training
def test(model, data_loader, loss_fn, device, use_tqdm):
    model.eval()
    current_loss = 0.0

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        # leave=False helps with screen clutter
        for images, labels in tqdm(data_loader, leave=False, disable=not use_tqdm):
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
    MODEL_NAME = "convnext_tiny"
    HP_INDEX = "set_3"
    HP_SET = HYPER_PARAMS[MODEL_NAME][HP_INDEX]
    USE_TQDM = True

    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = HP_SET["batch_size"]
    use_checkpointing = HP_SET["use_grad_checkpoint"]

    data_splits_df = pd.read_csv(CSV_FILE)
    test_split_df = pd.read_csv(CSV_TRAIN)

    city_names = sorted([name for name in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, name))])

    print(f"Performing Final Train on {MODEL_NAME} on {HP_INDEX}.\n")

    # Training and Testing Data
    train_loader = DataLoader(
        CitiesDataset(data_splits_df, DATA_ROOT, transform=get_transforms(is_train=True)),
        batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        CitiesDataset(test_split_df, TEST_ROOT, transform=get_transforms(is_train=False)),
        batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, drop_last=False
    )

    # Create the model for this fold
    model = get_model(MODEL_NAME, len(city_names), freeze_backbone=True, grad_checkpointing=use_checkpointing).to(device)
    loss_func = nn.CrossEntropyLoss(label_smoothing=HP_SET["label_smoothing"])
    scaler = GradScaler()

    # Results for this fold
    train_result = {"training_loss": [], "testing_loss": [], "final_predictions": [], "final_targets": []}

    # Training of classifier (if scheduled)
    # AdamW suggested for better generalization which seems applicable to our diverse dataset and task
    if HP_SET["classifier_epochs"] > 0:
        optimizer = optim.AdamW(model.get_classifier().parameters(), lr=HP_SET["lr_classifier"], weight_decay=HP_SET["weight_decay"])
        for epoch in range(HP_SET["classifier_epochs"]):
            training_loss = train_one_epoch(model, train_loader, optimizer, loss_func, device, scaler, USE_TQDM)
            # Ignore the accuracy while just training classifier
            train_result["training_loss"].append(training_loss)

            print(f"Completed classifier train epoch {epoch + 1}/{HP_SET['classifier_epochs']}.\n\tTraining loss: {training_loss}\n")

    if HP_SET["finetune_epochs"] > 0:
        model = unfreeze_model(model)
        # Splitting learing rates so new classifier keeps learning at faster rate while making smaller changes to whole network
        classifier_params = [param for name, param in model.named_parameters() if "classifier" in name]
        backbone_params   = [param for name, param in model.named_parameters() if "classifier" not in name]

        optimizer = optim.AdamW([
            {"params": classifier_params, "lr": HP_SET["lr_classifier"]},
            {"params": backbone_params, "lr": HP_SET["lr_finetune"]},
        ], weight_decay=HP_SET["weight_decay"])

        for epoch in range(HP_SET["finetune_epochs"]):
            training_loss = train_one_epoch(model, train_loader, optimizer, loss_func, device, scaler, USE_TQDM)

            train_result["training_loss"].append(training_loss)

            print(f"Completed finetune epoch {epoch + 1}/{HP_SET['finetune_epochs']}.\n\tTraining loss: {training_loss}\n")

    # Final Testing
    test_loss, predictions, targets = test(model, test_loader, loss_func, device, USE_TQDM)
    train_result["final_predictions"] = predictions
    train_result["final_targets"] = targets
    train_result["testing_loss"].append(test_loss)

    # Generate final reports for the model and hyperparameter set
    plot_test(train_result["training_loss"], train_result["testing_loss"], MODEL_NAME, HP_INDEX, RESULTS_DIR)
    confusion_matrix_gen(train_result["final_predictions"], train_result["final_targets"], MODEL_NAME, HP_INDEX, city_names, RESULTS_DIR)
    accuracy_f1_metrics(train_result["final_predictions"], train_result["final_targets"], city_names, MODEL_NAME, HP_INDEX, RESULTS_DIR)

    print(f"5-Fold Validation run on {MODEL_NAME} and {HP_INDEX} complete.")

if __name__ == "__main__":
    main()
