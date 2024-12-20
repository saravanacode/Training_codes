import torch
from torch import nn
from torchinfo import summary
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import random
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import json, math
import torchvision
from torch.nn import functional as F
import torch
from torch.utils.tensorboard import SummaryWriter

# Constants
BATCH_SIZE = 1024
IMG_SIZE = 224
SEED = 42
# Parameters
embed_dim = 768
hidden_dim = 3072
num_heads = 12
num_layers = 12
dropout = 0.0
num_classes = 4  # Update with actual number of classes
experiment_name = "vision_transformer_training" # Replace this
import comet_ml
from comet_ml import Experiment 
experiment = Experiment("Hy93O9e9gbVQ0eN1w0lGc2vAA")


comet_ml.login(project_name="rt-detr_model_train")


# Helper function for reproducible results
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# Define the data directory
data_dir = Path("/home/ubuntu/data_curation/AC_Classify")  # Replace this with the appropriate path.

# Define data transformations
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create the training dataset using ImageFolder
training_dataset = ImageFolder(root=data_dir / "train", transform=train_transform)

# Create the test dataset using ImageFolder
test_dataset = ImageFolder(root=data_dir / "test", transform=test_transform)

# Create the training dataloader using DataLoader
training_dataloader = DataLoader(
    dataset=training_dataset,
    shuffle=True,
    batch_size=BATCH_SIZE,
    num_workers=8,  # increase number of workers since we have multiple GPUs
    pin_memory=True
)

# Create the test dataloader using DataLoader
test_dataloader = DataLoader(
    dataset=test_dataset,
    shuffle=False,
    batch_size=BATCH_SIZE,
    num_workers=8,  # increase number of workers since we have multiple GPUs
    pin_memory=True
)

class CreatePatches(nn.Module):
    def __init__(
        self, channels=3, embed_dim=embed_dim, patch_size=16
    ):
        super().__init__()
        self.patch = nn.Conv2d(
            in_channels=channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # Flatten along dim = 2 to maintain channel dimension.
        patches = self.patch(x).flatten(2).transpose(1, 2)
        return patches
    
class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        super().__init__()

        self.pre_norm = nn.LayerNorm(embed_dim, eps=1e-06)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-06)
        self.MLP = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_norm = self.pre_norm(x)
        # MultiheadAttention returns attention output and weights,
        # we need only the outputs, so [0] index.
        x = x + self.attention(x_norm, x_norm, x_norm)[0]
        x = x + self.MLP(self.norm(x))
        return x
    
class ViT(nn.Module):
    def __init__(
        self, 
        img_size=IMG_SIZE,
        in_channels=3,
        patch_size=16,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=num_classes
    ):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size//patch_size) ** 2
        self.patches = CreatePatches(
            channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size
        )

        # Postional encoding.
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.attn_layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.attn_layers.append(
                AttentionBlock(embed_dim, hidden_dim, num_heads, dropout)
            )
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(embed_dim, eps=1e-06)
        self.head = nn.Linear(embed_dim, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, output_attentions=False):
        x = self.patches(x)
        b, n, _ = x.shape
 
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding

        x = self.dropout(x)
        attention_maps = []
        for layer in self.attn_layers:
            if output_attentions:
                 x, attention_map = layer.attention(x, x, x) # Output attention maps of current layer
                 attention_maps.append(attention_map)
            else:
                x = layer(x)
        x = self.ln(x)
        x = x[:, 0]
        logits = self.head(x)
        if output_attentions:
             return logits, attention_maps
        return logits

# --- Utility Functions ---
def save_experiment(experiment_name, config, model, train_losses, test_losses, accuracies, base_dir="experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)

    # Save the config
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'w') as f:
        json.dump(config, f, sort_keys=True, indent=4)

    # Save the metrics
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'w') as f:
        data = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'accuracies': accuracies,
        }
        json.dump(data, f, sort_keys=True, indent=4)

    # Save the model
    save_checkpoint(experiment_name, model, "final", base_dir=base_dir)


def save_checkpoint(experiment_name, model, epoch, base_dir="experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    cpfile = os.path.join(outdir, f'model_{epoch}.pt')
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), cpfile)
    else:
        torch.save(model.state_dict(), cpfile)


def load_experiment(experiment_name, checkpoint_name="model_final.pt", base_dir="experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    # Load the config
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'r') as f:
        config = json.load(f)
    # Load the metrics
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'r') as f:
        data = json.load(f)
    train_losses = data['train_losses']
    test_losses = data['test_losses']
    accuracies = data['accuracies']
    # Load the model
    model = ViT(
        img_size=config["IMG_SIZE"],
        in_channels=3,
        patch_size=config["patch_size"],
        embed_dim=config["embed_dim"],
        hidden_dim=config["hidden_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        num_classes=config["num_classes"]
    )
    cpfile = os.path.join(outdir, checkpoint_name)
    model.load_state_dict(torch.load(cpfile))
    return config, model, train_losses, test_losses, accuracies







def train_model(model, train_dataloader, test_dataloader, optimizer, criterion, epochs, device, config):
    model.to(device)
    if torch.cuda.device_count() > 1:  # If there are multiple GPUs
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)  # Use DataParallel to utilize multiple GPUs.

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_accuracies = []

    # Initialize TensorBoard writer
    writer = SummaryWriter()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} Train"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)
        print(f"Epoch {epoch+1}/{epochs} Train Loss: {train_loss:.4f}")

        # Log training loss to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(test_dataloader, desc=f"Epoch {epoch+1}/{epochs} Validation"):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(test_dataloader)
        val_losses.append(val_loss)
        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)
        print(f"Epoch {epoch+1}/{epochs} Validation Loss: {val_loss:.4f} Validation Accuracy: {val_accuracy:.2f}%")

        # Log validation loss and accuracy to TensorBoard
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(experiment_name, model, "best", base_dir="experiments")

    # Close the TensorBoard writer
    writer.close()

    return train_losses, val_losses, val_accuracies

if __name__ == '__main__':
    set_seed()
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define the config dictionary
    config = {
    "BATCH_SIZE" : BATCH_SIZE,
    "IMG_SIZE" : IMG_SIZE,
    "SEED" : SEED,
    "embed_dim" : embed_dim,
    "hidden_dim" : hidden_dim,
    "num_heads" : num_heads,
    "num_layers" : num_layers,
    "dropout" : dropout,
    "num_classes" : num_classes,
    "patch_size" : 16
    }


    model = ViT()
    print(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    rnd_int = torch.randn(1, 3, 224, 224)
    output = model(rnd_int)
    print(f"Output shape from model: {output.shape}")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Train the model
    train_losses, val_losses, val_accuracies = train_model(
        model, training_dataloader, test_dataloader, optimizer, criterion, epochs=50, device=device, config = config
    )
    print("Training complete.")


    # Save the experiment
    save_experiment(experiment_name, config, model, train_losses, val_losses, val_accuracies, base_dir="experiments")

    # Plotting
    epochs = range(1, 20 + 1)
    plt.figure(figsize=(12, 5))

    # Plotting training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plotting Validation Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
