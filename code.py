import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image, UnidentifiedImageError
import pandas as pd
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define the Dataset class
class FakeNewsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, image_transform, max_len=128):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = str(self.data.iloc[index]['Statement'])
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        image_path = self.data.iloc[index]['Image']
        if isinstance(image_path, str) and os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert('RGB')
                image = self.image_transform(image)
            except (UnidentifiedImageError, OSError):
                image = torch.zeros((3, 224, 224))
        else:
            image = torch.zeros((3, 224, 224))

        metadata = torch.tensor([
            hash(str(self.data.iloc[index]['Web'])) % 100,
            hash(str(self.data.iloc[index]['Category'])) % 100
        ], dtype=torch.float32)

        label = torch.tensor(1 if self.data.iloc[index]['Label'] == "TRUE" else 0, dtype=torch.float32)

        return input_ids, attention_mask, image, metadata, label

# Define the MF-BERT model
class MFBERTModel(nn.Module):
    def __init__(self):
        super(MFBERTModel, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.text_fc = nn.Linear(768, 256)

        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 256)

        self.metadata_fc = nn.Linear(2, 64)

        self.fusion_fc = nn.Linear(256 + 256 + 64, 128)
        self.classifier = nn.Linear(128, 1)

        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask, image, metadata):
        text_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = self.text_fc(text_outputs.pooler_output)
        image_features = self.cnn(image)
        metadata_features = self.metadata_fc(metadata)

        fused_features = torch.cat((text_features, image_features, metadata_features), dim=1)
        fused_features = self.dropout(self.fusion_fc(fused_features))

        return self.classifier(fused_features)  # No Sigmoid (Handled in Loss)

# Training function
def train_model(model, dataloader, criterion, optimizer, scaler, device, epoch, total_epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}/{total_epochs}")

    for batch in progress_bar:
        input_ids, attention_mask, image, metadata, labels = [x.to(device) for x in batch]
        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda"):
            outputs = model(input_ids, attention_mask, image, metadata).squeeze()
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=total_loss / len(dataloader))

    return total_loss / len(dataloader)

# Evaluation function with precision, recall, and F1-score
def evaluate_model(model, dataloader, criterion, device, epoch, total_epochs):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc=f"Evaluating Epoch {epoch+1}/{total_epochs}")

    with torch.no_grad():
        for batch in progress_bar:
            input_ids, attention_mask, image, metadata, labels = [x.to(device) for x in batch]
            outputs = model(input_ids, attention_mask, image, metadata).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predictions = (outputs > 0).float().cpu().numpy()
            labels = labels.cpu().numpy()

            all_predictions.extend(predictions)
            all_labels.extend(labels)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)

    print(f"Epoch {epoch+1}/{total_epochs} | Val Loss: {total_loss / len(dataloader):.4f} | "
          f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-score: {f1:.4f}")

    return total_loss / len(dataloader), accuracy, precision, recall, f1

# Main training pipeline
def main():
    df = pd.read_csv("/content/updated_dataset (1).csv")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    train_dataset = FakeNewsDataset(train_df, tokenizer, image_transform)
    test_dataset = FakeNewsDataset(test_df, tokenizer, image_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MFBERTModel().to(device)
    criterion = nn.BCEWithLogitsLoss()  
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # AMP Fix
    scaler = torch.amp.GradScaler()

    epochs = 1
    for epoch in range(epochs):
        train_loss = train_model(model, train_dataloader, criterion, optimizer, scaler, device, epoch, epochs)
        val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate_model(
            model, test_dataloader, criterion, device, epoch, epochs
        )
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Accuracy: {val_accuracy:.4f} | "
              f"Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1-score: {val_f1:.4f}")

if __name__ == "__main__":
    main()
