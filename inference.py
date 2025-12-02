import torch
import timm
from PIL import Image, ImageTk
from torchvision import transforms
from torchvision.transforms import v2
import tkinter as tk
from tkinter import Label, Button, filedialog

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#ViT Preprocessing
vit_transform = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                         std=(0.5, 0.5, 0.5))
])

#CNN Preprocessing
def resize_and_pad(img: Image.Image, target_size=(320, 320)):
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    new_img = Image.new("RGB", target_size, (0, 0, 0))
    left = (target_size[0] - img.size[0]) // 2
    top = (target_size[1] - img.size[1]) // 2
    new_img.paste(img, (left, top))
    return new_img

cnn_transform = v2.Compose([
    v2.Lambda(lambda img: resize_and_pad(img)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
])

class_names = [
    "Calgary", "Charlottetown", "Edmonton", "Halifax", "Hamilton",
    "Kitchener-Waterloo", "Montreal", "Ottawa-Gatineau", "Quebec City",
    "Saskatoon", "St Johns", "Toronto", "Vancouver", "Victoria", "Winnipeg"
]

#ViT Model
model1 = timm.create_model("swinv2_base_window12_192", pretrained=False, num_classes=15)
model1.load_state_dict(torch.load("vit_model/swinv2_base_window12_192_0_finetuned_canadian_streetview.bin", map_location=device))
model1.to(device)
model1.eval()

#CNN Model
model2 = timm.create_model(
    "convnext_tiny",
    pretrained=False,
    num_classes=15
)
checkpoint = torch.load(
    "cnn_model/convnext_tiny_set_3_final.bin",
    map_location=device
)
model2.load_state_dict(checkpoint['model_state_dict'])
model2.to(device)
model2.eval()

#Make prediction
def predict(img_path: str, model: torch.nn.Module, transform) -> str:
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        pred_idx = logits.argmax(dim=1).item()
    return class_names[pred_idx]

#GUI
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Canadian StreetView Classifier")
        self.root.geometry("550x700")

        self.current_image_path = None

        self.image_frame = tk.Frame(root, width=500,  height=350, bg="gray")
        self.image_frame.pack_propagate(False) 
        self.image_frame.place(relx=0.5, y=10, anchor='n')

        self.img_label = Label(self.image_frame, text="No image loaded", bg="gray", fg="white") 
        self.img_label.pack(expand=True, fill='both') 
        self.pred_label1 = Label(root, text="", font=("Arial", 14))
        self.pred_label2 = Label(root, text="", font=("Arial", 14))
        self.pred_label1.place(relx=0.5, y=400, anchor='center') 
        self.pred_label2.place(relx=0.5, y=430, anchor='center') 
        
        Button(root, text="Upload Image", command=self.upload_image).place(relx=0.5, y=500, anchor='center')
        Button(root, text="Reveal Prediction", command=self.reveal_prediction).place(relx=0.5, y=550, anchor='center')

    def display_image(self, path):
        img = Image.open(path)
        img = img.resize((500, 350), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(img)

        self.img_label.config(image=self.tk_img, text="", width=500, height=350)
        self.img_label.image = self.tk_img 
        self.pred_label1.config(text="")
        self.pred_label2.config(text="")

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
        if not file_path:
            return
        self.current_image_path = file_path
        self.display_image(file_path)

    def reveal_prediction(self):
        if not self.current_image_path:
            self.pred_label1.config(text="Upload an image first.")
            self.pred_label2.config(text="")
            return

        pred1 = predict(self.current_image_path, model1, vit_transform)
        pred2 = predict(self.current_image_path, model2, cnn_transform)
        self.pred_label1.config(text=f"ViT Prediction: {pred1}")
        self.pred_label2.config(text=f"CNN Prediction: {pred2}")

root = tk.Tk()
app = App(root)
root.mainloop()
