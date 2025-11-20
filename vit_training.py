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

num_workers = 4
batch_size = 16
use_amp = True
drop_last = True
pin_memory = True

#Load dataset
dataset = load_dataset("SABR22/Canadian-streetview-cities", streaming=False)
train_ds = dataset["train"]
test_ds  = dataset["test"]

#Resize each image to 224 x 224
IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                         std=(0.5, 0.5, 0.5))
])

#Apply transforms and stack
def collate_fn(batch):
    images = [transform(item["image"].copy()) for item in batch]
    labels = [item["label"] for item in batch]
    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels

#Main training function
def train_model(model, train_data, val_data=None, epochs=5, batch_size=32, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True, drop_last=True)
    if val_data:
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True, drop_last=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = GradScaler() if use_amp else None

    #Store loss
    train_loss_list = []
    val_loss_list = []

    for epoch in range(epochs):

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

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            loader.set_postfix(loss=running_loss / (step * batch_size))
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
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item() * imgs.size(0)

                    preds = outputs.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            epoch_val_loss = val_running_loss / len(val_loader.dataset)
            val_loss_list.append(epoch_val_loss)
            val_acc = correct / total
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | " f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f}")
    
    return train_loss_list, val_loss_list

#K-fold validation
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 15

avg_train_loss = None
avg_val_loss = None
final_targets = []
final_predictions = []

fold_results = []
indices = range(len(train_ds))

for fold, (train_i, val_i) in enumerate(kf.split(indices), 1):
    print(f"Fold {fold}")

    #Train on fold i
    train_subset = train_ds.select(train_i)
    val_subset   = train_ds.select(val_i)

    model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=num_classes)
    model.to(device)

    train_loss, val_loss = train_model(model, train_subset, val_subset, epochs=6, device=device)

    if avg_train_loss is None:
        avg_train_loss = train_loss.copy()
        avg_val_loss = val_loss.copy()
    else:
        for i in range(len(train_loss)):
            avg_train_loss[i] += train_loss[i]
            avg_val_loss[i] += val_loss[i]

    #Get prediction on fold i
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True, drop_last=False)
    model.eval()
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            final_targets.extend(labels.cpu().tolist())
            final_predictions.extend(preds.cpu().tolist())

avg_train_loss = [x / k for x in avg_train_loss]
avg_val_loss = [x / k for x in avg_val_loss]

print("Avg Train Loss per Epoch:", avg_train_loss)
print("Avg Val Loss per Epoch:", avg_val_loss)

#Save results
results_dict = {
    "avg_train_loss": avg_train_loss,
    "avg_val_loss": avg_val_loss,
    "final_targets": final_targets,
    "final_predictions": final_predictions
}

# Save to JSON
with open("vit_results.json", "w") as f:
    json.dump(results_dict, f)

print("Results saved to vit_results.json")

#Train on full set
final_model = timm.create_model('vit_small_patch16_224', pretrained=True)
final_model.head = nn.Linear(final_model.head.in_features, num_classes)
final_model.to(device)
train_model(final_model, train_ds, epochs=12, batch_size=batch_size, device=device)

#Test on test set
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True, drop_last=False)
final_model.eval()
correct, total = 0, 0
loader = tqdm(test_loader, desc="Testing", leave=True)
with torch.no_grad():
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = final_model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        loader.set_postfix({"acc": f"{correct/total:.4f}"})
test_acc = correct / total
print(f"Final Test Accuracy: {test_acc:.4f}")

#Save model
torch.save(final_model.state_dict(), "vit_finetuned_canadian_streetview.pth")
print("Model saved to vit_finetuned_canadian_streetview.pth")