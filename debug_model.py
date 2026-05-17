import sys
from pathlib import Path
sys.path.append('tasks/task2-detector/src')
import torch
import torchvision
from torchvision import transforms
from train import MNISTClassifier
from model import load_mnist_model

model = load_mnist_model('tasks/task2-detector/models/mnist_classifier.npz')

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])
dataset=torchvision.datasets.MNIST(root='tasks/task2-detector/data',train=False,download=True,transform=transform)

correct = 0
for i in range(100):
    img, label = dataset[i]
    pred = model(img.unsqueeze(0)).argmax(1).item()
    if pred == label:
        correct += 1
print(f"Accuracy on first 100 test images: {correct}/100")
