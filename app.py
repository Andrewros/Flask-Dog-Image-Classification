from flask import Flask, render_template, redirect, request
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
app=Flask(__name__)
CLASSES = ['affenpinscher', 'akita', 'corgi']
preprocessedsteps=transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize([32, 32]),
    transforms.ToTensor()
])
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Conv2d(1, 6, 3),#6x30x30
            nn.ReLU(),
            nn.Conv2d(6,16, 3), #16x28x28
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.Linear(16*14*14, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
    def forward(self, x):
        return self.layers(x)
model=ImageClassifier()
model.load_state_dict(torch.load("model.pt"))
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method=="GET":
        return render_template("index.html")
    elif request.method=="POST":
        file=request.files['image']
        img = Image.open(file.stream)
        img = preprocessedsteps(img).unsqueeze(0)
        with torch.no_grad():
            model.eval()
            outputs = model(img)
            predicted = torch.max(outputs, 1).indices
        predicted_class = CLASSES[predicted]
        return f"I believe that is a picture of {predicted_class}"
        
if __name__=="__main__":
    app.run(port=3000, debug=True)