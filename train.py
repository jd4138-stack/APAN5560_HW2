
import torch, torch.nn as nn, torch.optim as optim
from helper_lib.data_loader import get_cifar10_loaders
from helper_lib.model import get_model
from helper_lib.trainer import train_model
from helper_lib.evaluator import evaluate_model

def main(device="cuda" if torch.cuda.is_available() else "cpu"):
    train_loader, val_loader, test_loader = get_cifar10_loaders(image_size=64, batch_size=128)
    model = get_model("CNN", num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model = train_model(model, train_loader, val_loader, criterion, optimizer,
                        device=device, epochs=10, checkpoint_dir="checkpoints")

    loss, acc = evaluate_model(model, test_loader, criterion, device=device)
    print(f"Test Loss {loss:.4f} | Test Acc {acc*100:.2f}%")

if __name__ == "__main__":
    main()
