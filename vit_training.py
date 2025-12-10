import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.amp import GradScaler, autocast
from datasets import load_dataset
import timm
from sklearn.model_selection import KFold
from tqdm.auto import tqdm
import json
import time
import sys
import os

#Choose model
models_dict = {
    'vit_base_patch16_224': 224,
    'deit3_base_patch16_224': 224,
    'swinv2_base_window12_192': 192
}

if len(sys.argv) < 3:
    print(f"Usage: python {sys.argv[0]} <model_index> <param_set_index>")
    sys.exit(1)

model_index = int(sys.argv[1])
param_set_index = int(sys.argv[2])

model_names = list(models_dict.keys())
if model_index < 0 or model_index >= len(model_names):
    print(f"Invalid model index. Must be between 0 and {len(model_names)-1}")
    sys.exit(1)

if param_set_index not in [0, 1]:
    print("Invalid param set index. Must be 0 or 1")
    sys.exit(1)

model_name = model_names[model_index]
IMG_SIZE = models_dict[model_name]
print(f"Using model: {model_name}")

#ViT Sets [LR, WD, BS, Epoch]
model_params = {
    0: [[1e-4, 1e-4, 32, 12], [2e-4, 1e-4, 64, 18]],   #ViT
    1: [[5e-5, 1e-4, 32, 12], [5e-5, 0.05, 64, 18]],   #DeiT
    2: [[2e-5, 5e-4, 34, 18], [3e-5, 5e-4, 32, 18]]    #Swin
}

#Hyperparams
lr, wd, batch_size, final_epochs = model_params[model_index][param_set_index]
num_workers = 8
num_classes = 15
fold_epochs = 8
use_amp = True
print(f"Hyperparameters selected: LR={lr}, WD={wd}, Batch Size={batch_size}, Epochs={final_epochs}, Num Workers={num_workers}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load dataset
dataset = load_dataset("SABR22/Canadian-streetview-cities", streaming=False)
train_ds = dataset["train"]
test_ds  = dataset["test"]

# Data augmentation for training
train_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.crop((0, int(0.05 * img.height), img.width, int(0.85 * img.height)))),  # Crop top 5%, bottom 15%
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.15, p=0.5)], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Simpler test transform
test_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.crop((0, int(0.05 * img.height), img.width, int(0.85 * img.height)))),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

#Apply transforms and stack
def train_collate_fn(batch):
    images = []
    labels = []
    for item in batch:
        img_pil = item["image"].copy().convert("RGB")
        img_np = np.array(img_pil)
        transformed = train_transform(image=img_np)
        images.append(transformed['image']) 
        labels.append(item["label"])
    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels

def test_collate_fn(batch):
    images = []
    labels = []
    for item in batch:
        img_pil = item["image"].copy().convert("RGB")
        img_np = np.array(img_pil)
        transformed = test_transform(image=img_np)
        images.append(transformed['image'])
        labels.append(item["label"])
    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels

#Main training function
def train_model(model, train_data, val_data=None, epochs=5, batch_size=16, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=train_collate_fn, pin_memory=True, drop_last=True)
    if val_data:
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=test_collate_fn, pin_memory=True, drop_last=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scaler = GradScaler() if use_amp else None

    #Warm up and scheduler
    warmup_epochs = 2
    cosine_epochs = epochs - warmup_epochs

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs
    )

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=lr * 0.01
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    #Store loss
    train_loss_list = []
    val_loss_list = []

    #Iterate over epochs
    for epoch in range(epochs):
        epoch_start_time = time.time()

        #Training
        model.train()
        running_loss = 0.0
        total_samples = 0
        loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for step, (imgs, labels) in enumerate(loader, 1):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            if use_amp:
                with autocast(device_type=device.type):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            total_samples += imgs.size(0)
            loader.set_postfix(loss=running_loss / (step * batch_size))
        epoch_train_loss = running_loss / total_samples if total_samples > 0 else 0
        train_loss_list.append(epoch_train_loss)

        #Validation
        val_acc = None
        if val_data:
            model.eval()
            correct, total = 0, 0
            val_running_loss = 0.0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item() * imgs.size(0)

                    preds = outputs.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            epoch_val_loss = val_running_loss / len(val_loader.dataset)
            val_loss_list.append(epoch_val_loss)
            val_acc = correct / total
            
            epoch_end_time = time.time()
            epoch_total_time = epoch_end_time - epoch_start_time
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | " f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_acc:.4f} | Completed in {epoch_total_time/60:.2f} minutes")
        else:
            epoch_end_time = time.time()
            epoch_total_time = epoch_end_time - epoch_start_time
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Completed in {epoch_total_time/60:.2f} minutes")

        scheduler.step()

    return train_loss_list, val_loss_list

#K-fold validation
def kfold_train(train_ds, model_name, num_classes=15, k=5, epochs=fold_epochs, batch_size=batch_size, device=None):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []
    indices = range(len(train_ds))
    kfold_start = time.time()

    #Iterate over folds
    for fold, (train_i, val_i) in enumerate(kf.split(indices), 1):
        fold_start_time = time.time()
        print(f"Starting fold {fold}")

        #Train on fold i
        train_subset = train_ds.select(train_i)
        val_subset   = train_ds.select(val_i)

        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        model.to(device)
        train_loss, val_loss = train_model(model, train_subset, val_subset, epochs=epochs, batch_size=batch_size, device=device)

        fold_dict = {
            "final_predictions": [],
            "final_targets": [],
            "training_loss": train_loss,
            "validation_loss": val_loss
        }

        #Get prediction on fold i
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=test_collate_fn, pin_memory=True, drop_last=False)
        model.eval()
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                fold_dict["final_targets"].extend(labels.cpu().tolist())
                fold_dict["final_predictions"].extend(preds.cpu().tolist())

        fold_results.append(fold_dict)

        fold_end_time = time.time()
        fold_total_time = fold_end_time - fold_start_time
        print(f"Fold {fold} completed in {fold_total_time/60:.2f} minutes")

    kfold_end = time.time()
    kfold_total = kfold_end - kfold_start
    print(f"All {k} folds completed in {kfold_total/60:.2f} minutes")

    # Save to JSON
    os.makedirs("vit_results", exist_ok=True)
    out_path = os.path.join("vit_results", f"{model_name}_results_{param_set_index}.json")
    with open(out_path, "w") as f:
        json.dump(fold_results, f)
    print(f"Results saved to vit_results/{model_name}_results_{param_set_index}.json")

#Kfol_train
#kfold_train(train_ds, model_name, num_classes=num_classes, k=5, epochs=fold_epochs, batch_size=batch_size, device=device)

#Train on full set
final_model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
final_model.to(device)

final_start_time = time.time()
final_train_losses, _ = train_model(final_model, train_ds, val_data=None, epochs=final_epochs, batch_size=batch_size, device=device)
final_end_time = time.time()
final_time = final_end_time - final_start_time
print(f"Final training completed in {final_time/60:.2f} minutes")

#Test on test set
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=test_collate_fn, pin_memory=False, drop_last=False)
final_model.eval()
final_predictions = []
final_targets = []
correct, total = 0, 0

print("Evaluating on official test set...")
with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Final Test"):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = final_model(imgs)
        preds = outputs.argmax(dim=1)
        
        final_predictions.extend(preds.cpu().tolist())
        final_targets.extend(labels.cpu().tolist())
        
        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_acc = correct / total
print(f"Final Test Accuracy: {test_acc:.4f}")

final_fold_dict = {
    "final_predictions": final_predictions,
    "final_targets": final_targets,
    "training_loss": final_train_losses,
    "validation_loss": []
}
final_results = [final_fold_dict]

os.makedirs("vit_results", exist_ok=True)
final_json_path = f"vit_results/{model_name}_final_training_results_{param_set_index}_V2.json"
with open(final_json_path, "w") as f:
    json.dump(final_results, f, indent=2)
print(f"Final results saved → {final_json_path}")

#Save model
os.makedirs("models", exist_ok=True)
torch.save(final_model.state_dict(), f"models/{model_name}_{param_set_index}_finetuned_canadian_streetview_V2.pth")
print(f"Model saved to models/{model_name}_{param_set_index}_finetuned_canadian_streetview_V2.pth")