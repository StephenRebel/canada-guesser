import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.amp import GradScaler, autocast
from datasets import load_dataset
import timm
from timm.data import Mixup, FastCollateMixup
from timm.data import create_transform
from timm.scheduler import CosineLRScheduler
from torch.utils.data._utils.collate import default_collate
from sklearn.model_selection import KFold
from tqdm.auto import tqdm
import json
import time
import sys
import os

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

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
    1: [[5e-5, 1e-4, 32, 12], [3e-5, 1e-4, 48, 18]],   #DeiT
    2: [[5e-5, 1e-4, 24, 12], [3e-5, 5e-4, 32, 18]]    #Swin
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

#Data augmentation
train_transform = create_transform(
    input_size=IMG_SIZE,
    is_training=True,
    color_jitter=0.4,
    auto_augment='rand-m9-mstd0.5-inc1',
    interpolation='bicubic'
)

val_transform = create_transform(
    input_size=IMG_SIZE,
    is_training=False,
    interpolation='bicubic',
)

mixup_args = dict(
    mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
    prob=1.0, switch_prob=0.5, mode='batch',
    correct_lam=True, label_smoothing=0.1, num_classes=num_classes
)
mixup_fn = FastCollateMixup(**mixup_args)

#Collate functions
def train_collate_fn(batch):
    processed = []
    for item in batch:
        img = train_transform(item["image"])
        label = item["label"]
        if isinstance(label, torch.Tensor):
            label = label.item()
        elif not isinstance(label, int):
            label = int(label)
        processed.append((img, label))

    return mixup_fn(*default_collate(processed))

def val_collate_fn(batch):
    images = []
    labels = []
    for item in batch:
        images.append(val_transform(item["image"].copy()))
        label = item["label"]
        if isinstance(label, torch.Tensor):
            label = label.item()
        elif not isinstance(label, int):
            label = int(label)
        labels.append(label)
    
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)

#Main training function
def train_model(model, train_data, val_data=None, epochs=5, batch_size=16, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=train_collate_fn, pin_memory=True, drop_last=True)
    if val_data:
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=val_collate_fn, pin_memory=True, drop_last=False)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scaler = GradScaler() if use_amp else None

    #Warm up and scheduler
    total_steps = len(train_loader) * epochs
    warmup_steps = max(500, int(0.1 * total_steps))
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=total_steps - warmup_steps,
        lr_min=1e-6,
        warmup_lr_init=1e-6,
        warmup_t=warmup_steps,
        cycle_limit=1
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
        loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for step, (imgs, labels) in enumerate(loader, 1):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            with autocast(device_type=device.type):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item() * imgs.size(0)
            current_lr = optimizer.param_groups[0]['lr']
            loader.set_postfix(loss=running_loss/(step*batch_size), lr=f"{current_lr:.2e}")

        epoch_train_loss = running_loss / len(train_loader.dataset)
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
                    with autocast(device_type=device.type):
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
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=val_collate_fn, pin_memory=True, drop_last=False)
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


# #Kfol_train
kfold_train(train_ds, model_name, num_classes=num_classes, k=5, epochs=fold_epochs, batch_size=batch_size, device=device)

# #Train on full set
# final_model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
# final_model.to(device)

# final_start_time = time.time()
# train_model(final_model, train_ds, epochs=final_epochs, batch_size=batch_size, device=device)
# final_end_time = time.time()
# final_time = final_end_time - final_start_time
# print(f"Final training completed in {final_time/60:.2f} minutes")

# #Test on test set
# test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=val_collate_fn, pin_memory=False, drop_last=False)
# final_model.eval()
# correct, total = 0, 0
# loader = tqdm(test_loader, desc="Testing", leave=True)
# with torch.no_grad():
#     for imgs, labels in loader:
#         imgs, labels = imgs.to(device), labels.to(device)
        
#         with autocast(device_type=device.type):
#             logits = final_model(imgs)
#         preds = logits.argmax(dim=1)
        
#         correct += (preds == labels).sum().item()
#         total += labels.size(0)
#         loader.set_postfix(acc=f"{correct/total:.4f}")
# test_acc = correct / total
# print(f"Final Test Accuracy: {test_acc:.4f}")

# #Save model
# os.makedirs("models", exist_ok=True)
# torch.save(final_model.state_dict(), f"models/{model_name}_{param_set_index}_finetuned_canadian_streetview.pth")
# print(f"Model saved to models/{model_name}_{param_set_index}_finetuned_canadian_streetview.pth")
