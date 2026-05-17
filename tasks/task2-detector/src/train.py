"""Training scaffold for the Task 2 MNIST digit classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

TASK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MNIST_DATA_DIR = TASK_ROOT / "data"


def download_mnist_dataset(data_dir: Path = DEFAULT_MNIST_DATA_DIR) -> Path:
    """Download torchvision MNIST into the Task 2 data directory."""
    import torchvision

    data_dir.mkdir(parents=True, exist_ok=True)
    torchvision.datasets.MNIST(root=data_dir, train=True, download=True)
    torchvision.datasets.MNIST(root=data_dir, train=False, download=True)
    return data_dir / "MNIST"


class MNISTClassifier(nn.Module):
    """Small PyTorch classifier scaffold for 28x28 MNIST crops."""

    def __init__(self, input_size: int = 28 * 28, num_classes: int = 10) -> None:
        super().__init__()
        # TODO(student): fill in your custom model architectures
        #28*28展开然后两个FeedForward全连接层
        self.flatten=nn.Flatten()
        self.linear_relu_stack=nn.Sequential(
            nn.Linear(input_size,128),
            nn.ReLU(),
            nn.Linear(128,num_classes)
        )


    def forward(self, inputs):
        # TODO(student): fill in your forward process according to your model
        x=self.flatten(inputs)
        logits=self.linear_relu_stack(x)
        return logits


def select_training_device(torch_module) -> str:
    # TODO(student): Pick the best accelerator available on the student's PC.
    # if torch reports CUDA is available:
    #     return "cuda" for NVIDIA GPU training
    # else if torch reports MPS is available:
    #     return "mps" for Apple Silicon GPU training
    # otherwise:
    #     return "cpu" so training still works without an accelerator
    if torch_module.cuda.is_available():
        return "cuda"
    elif hasattr(torch_module.backends,"mps")and torch_module.backends.mps.is_available():
        return"mps"
    else:
        return "cpu"


def train_mnist_classifier(dataset_dir: Path, output_path: Path) -> Path:
    
    from torch.utils.data import DataLoader, random_split
    import torchvision
    from torchvision import transforms
    device=select_training_device(torch)
    print(f"Using device:{device}")
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])
    dataset=torchvision.datasets.MNIST(root=dataset_dir.parent,train=True,download=True,transform=transform)
    train_size=50000
    val_size=len(dataset)-train_size
    train_dataset,val_dataset=random_split(dataset,[train_size,val_size])
    train_loader=DataLoader(train_dataset,batch_size=64,shuffle=True)
    val_loader=DataLoader(val_dataset,batch_size=64,shuffle=False)
    
    model=MNISTClassifier().to(device)
    loss_fn=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    epochs=5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")

        model.train()
        for batch, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device) # 把数据丢进 GPU/CPU
            
            # 前向传播
            pred = model(X)
            loss = loss_fn(pred, y)
            
            # 反向传播与优化
            optimizer.zero_grad() # 清空上一次的梯度
            loss.backward()       # 反向传播计算新梯度
            optimizer.step()      # 更新模型权重
            
            if batch % 100 == 0:
                print(f"Loss: {loss.item():>7f}  [{batch * len(X):>5d}/{len(train_dataset):>5d}]")
        # 验证阶段（看模型有没有过拟合）
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad(): # 验证时不计算梯度，省显存
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        test_loss /= len(val_loader)
        correct /= len(val_dataset)
        print(f"Validation Error: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    # 7. 训练完毕，保存模型权重
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    return output_path
            



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Task 2 MNIST digit classifier.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_MNIST_DATA_DIR / "MNIST", help="Directory containing labeled MNIST board crops.")
    parser.add_argument("--output", type=Path, default=TASK_ROOT / "models" / "mnist_classifier.npz", help="Where to save the trained classifier.")
    parser.add_argument("--download-mnist", action="store_true", help="Download MNIST into tasks/task2-detector/data/MNIST before training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.download_mnist:
        dataset_path = download_mnist_dataset(DEFAULT_MNIST_DATA_DIR)
        print(f"Downloaded MNIST dataset to: {dataset_path}")
        return

    output_path = train_mnist_classifier(args.dataset_dir, args.output)
    print(f"Saved MNIST classifier to: {output_path}")


if __name__ == "__main__":
    main()
