from pathlib import Path
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
import accelerate
import transformers
import torch
from typing import List, Dict, Any
import comet_ml
from comet_ml import Experiment 
experiment = Experiment("Hy93O9e9gbVQ0eN1w0lGc2vAA")


comet_ml.login(project_name="vit_model_train")


# Define the data directory
data_dir = Path("/home/ubuntu/data_curation/AC_Classify")  # Replace this with the appropriate path.
BATCH_SIZE = 1024
IMG_SIZE = 224
# Define data transformations
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
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

# Extract class-to-index mapping
class_to_idx = training_dataset.class_to_idx

# Create id2label and label2id
id2label = {v: k for k, v in class_to_idx.items()}  # Reverse of class_to_idx
label2id = class_to_idx  # Already available

# Print the mappings
print("id2label:", id2label)
print("label2id:", label2id)



from transformers import ViTForImageClassification

model = ViTForImageClassification.from_pretrained('/home/ubuntu/data_curation/experiments/vision_transformer_training',
                                                  id2label=id2label,
                                                  label2id=label2id)

metric_name = "accuracy"

args = TrainingArguments(
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=512,  # Adjust batch size based on your available GPU memory
    per_device_eval_batch_size=512,  # Adjust batch size based on your available GPU memory
    num_train_epochs=150,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    logging_dir='logs',
    remove_unused_columns=False,
    output_dir="out_save",
    dataloader_num_workers=8,
    dataloader_pin_memory=True 
)

from sklearn.metrics import accuracy_score
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return dict(accuracy=accuracy_score(predictions, labels))
     

def custom_data_collator(features: List[tuple]) -> Dict[str, Any]:
    images = [item[0] for item in features]
    labels = [item[1] for item in features]
    batch = {"pixel_values": torch.stack(images), "labels": torch.tensor(labels)}
    return batch

import torch

trainer = Trainer(
    model,
    args,
    train_dataset=training_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    data_collator=custom_data_collator,
    
)


trainer.train()



outputs = trainer.predict(test_dataset)
print(outputs.metrics)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_true = outputs.label_ids
y_pred = outputs.predictions.argmax(1)

labels = training_dataset.classes
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(xticks_rotation=45)