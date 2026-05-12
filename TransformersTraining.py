import os
import torch
from tqdm import tqdm
import torch.nn as nn
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaConfig, RobertaModel, RobertaTokenizer,
    ElectraConfig, ElectraModel, ElectraTokenizer,
    LongformerConfig, LongformerModel, LongformerTokenizer,
    XLNetConfig, XLNetModel, XLNetTokenizer,
    get_scheduler
)
from torch.optim import AdamW
from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification, AutoModelForSequenceClassification, AutoModelForMaskedLM
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve


import matplotlib.pyplot as plt
try:
        import keyboard
except ImportError:
        keyboard = None

BATCH_SIZE = 16
MAX_LEN = 256
NUM_ITERS = 1000

import pandas as pd
def split_dataframe(df, train_perc=0.7, val_perc=0.05, test_perc=0.25):
    assert train_perc + val_perc + test_perc == 1.0, "Percentages must sum to 1.0"
    train_df = df.sample(frac=train_perc, random_state=42)
    remaining = df.drop(train_df.index)
    val_df = remaining.sample(frac=val_perc / (val_perc + test_perc), random_state=42)
    test_df = remaining.drop(val_df.index)
    return train_df, val_df, test_df


def prepare_ckpt_dir(path: str):
    os.makedirs(path, exist_ok=True)



class TextDataset(Dataset):

    def __init__(self, df, tokenizer):
        self.texts = df['cleaned_text'].astype(str).tolist()
        self.labels = df['source'].astype(float).tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=MAX_LEN,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.float).unsqueeze(0)
        }



def build_roberta(config_name: str = 'roberta-base') -> RobertaModel:
    config = RobertaConfig.from_pretrained(config_name)
    return RobertaModel(config)


def build_electra(config_name: str = 'google/electra-base-discriminator') -> ElectraModel:
    config = ElectraConfig.from_pretrained(config_name)
    return ElectraModel(config)

def build_longformer(config_name: str = 'allenai/longformer-base-4096') -> LongformerModel:
    config = LongformerConfig.from_pretrained(config_name)
    return LongformerModel(config)


def build_xlnet(config_name: str = 'xlnet-base-cased') -> XLNetModel:
    config = XLNetConfig.from_pretrained(config_name)
    return XLNetModel(config)


def build_arabert_classifier():
    model_name = "aubmindlab/bert-base-arabertv2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    return model, tokenizer

def build_araelectra_classifier():
    model_name = "aubmindlab/araelectra-base-generator"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    return model, tokenizer

def plot_confusion_matrix(cm, classes=('0','1')):
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment='center',
                     color='white' if cm[i, j] > thresh else 'black')
        plt.tight_layout()
        plt.show()





def prepare_ckpt_dir(ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)

def plot_confusion_matrix(cm):
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ['0', '1'])
    plt.yticks(tick_marks, ['0', '1'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def train_Transformers(model: nn.Module,
                       tokenizer,
                       train_df,
                       val_df,
                       lr: float,
                       weight_decay: float,
                       epochs: int,
                       loss_check: int,
                       checkpoint_dir: str = None):

    mtype = model.config.model_type if hasattr(model.config, 'model_type') else 'model'
    ckpt_dir = checkpoint_dir or f"checkpoints_{mtype}"
    prepare_ckpt_dir(ckpt_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hidden_size = model.config.hidden_size
    classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(hidden_size, 1)
    ).to(device)
    model = model.to(device)

    train_ds = TextDataset(train_df, tokenizer)
    val_ds = TextDataset(val_df, tokenizer)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

    optimizer = AdamW(list(model.parameters()) + list(classifier.parameters()),
                      lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    global_iter = 0
    best_val_loss = float('inf')

    iter_train_losses = []
    iter_val_losses = []
    iter_indices = []
    val_iter_indices = []

    stop_training = False

    for epoch in range(epochs):
        model.train()
        classifier.train()
        # Use tqdm for a progress bar over the training dataloader
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}")
        epoch_train_loss = 0.0

        for batch in pbar:
            if keyboard is not None and keyboard.is_pressed('q'):
                stop_training = True

            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=masks)
            cls_tok = outputs.last_hidden_state[:, 0, :]
            logits = classifier(cls_tok)
            loss = criterion(logits, labels.float())
            loss.backward()
            optimizer.step()

            global_iter += 1
            epoch_train_loss += loss.item()

            iter_indices.append(global_iter)
            iter_train_losses.append(loss.item())

            # Update the progress bar with current training loss
            pbar.set_postfix({'loss': loss.item()})

            if global_iter % NUM_ITERS == 0:
                path = os.path.join(ckpt_dir, f"ckpt_iter{global_iter}.pt")
                torch.save({
                    'epoch': epoch,
                    'iter': global_iter,
                    'model': model.state_dict(),
                    'clf': classifier.state_dict(),
                    'opt': optimizer.state_dict(),
                    'best_val_loss': best_val_loss
                }, path)
                print(f"\nSaved checkpoint at iter {global_iter}: {path}")
                plt.figure()
                plt.plot(iter_indices, iter_train_losses, label='Train Loss')
                if val_iter_indices:
                    plt.plot(val_iter_indices, iter_val_losses, label='Val Loss')
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.title(f'Loss Curve up to Iter {global_iter}')
                plt.legend()
                plt.savefig(os.path.join(ckpt_dir, f'loss_up_to_iter{global_iter}.png'))
                plt.close()

            if global_iter % loss_check == 0:
                model.eval()
                classifier.eval()
                val_loss = 0.0
                sampled = random.sample(list(val_dl), max(1, len(val_dl)//100))
                with torch.no_grad():
                    for vb in sampled:
                        vi = vb['input_ids'].to(device)
                        vm = vb['attention_mask'].to(device)
                        vl = vb['label'].to(device)
                        out = model(input_ids=vi, attention_mask=vm)
                        cls_v = out.last_hidden_state[:, 0, :]
                        logits_v = classifier(cls_v)
                        val_loss += criterion(logits_v, vl.float()).item()
                val_loss /= len(sampled)

                avg_train_loss = epoch_train_loss / loss_check
                pbar.set_postfix({'loss': loss.item(), 'avg_train_loss': avg_train_loss, 'val_loss': val_loss})
                epoch_train_loss = 0.0 # Reset for the next loss_check interval

                iter_val_losses.append(val_loss)
                val_iter_indices.append(global_iter)

                thresh = 0.01 if best_val_loss < 0.1 else 0.1
                if best_val_loss - val_loss >= thresh:
                    best_val_loss = val_loss
                    path = os.path.join(ckpt_dir, f"best_iter{global_iter}_val_{val_loss:.4f}.pt")
                    torch.save({
                        'epoch': epoch,
                        'iter': global_iter,
                        'model': model.state_dict(),
                        'clf': classifier.state_dict(),
                        'opt': optimizer.state_dict(),
                        'best_val_loss': best_val_loss
                    }, path)
                    print(f"\nSaved improved checkpoint: {path}") # Keep this print for important save notifications

                model.train()
                classifier.train()

            if stop_training:
                print("\nStopping training by user request...")
                break # Exit the inner batch loop

        if stop_training:
            # Plot the loss curve up to the point of stopping
            plt.figure()
            plt.plot(iter_indices, iter_train_losses, label='Train Loss')
            if val_iter_indices:
                plt.plot(val_iter_indices, iter_val_losses, label='Val Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title(f'Loss Curve up to Iter {global_iter} (Stopped)')
            plt.legend()
            plt.savefig(os.path.join(ckpt_dir, f'loss_curve_stopped_iter{global_iter}.png'))
            plt.close() # Close the plot to prevent it from displaying immediately if not desired
            print(f"Saved loss curve at stop: {os.path.join(ckpt_dir, f'loss_curve_stopped_iter{global_iter}.png')}")

            path = os.path.join(ckpt_dir, f'last_iter{global_iter}.pt')
            torch.save({
                'epoch': epoch,
                'iter': global_iter,
                'model': model.state_dict(),
                'clf': classifier.state_dict(),
                'opt': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, path)
            print(f"Stopped by user. Saved last checkpoint: {path}")
            break # Exit the outer epoch loop

        # Epoch-end validation (if not stopped)
        if not stop_training:
            model.eval()
            classifier.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for batch in val_dl:
                    ids = batch['input_ids'].to(device)
                    mask = batch['attention_mask'].to(device)
                    lbl = batch['label'].to(device)
                    out = model(input_ids=ids, attention_mask=mask)
                    cls_e = out.last_hidden_state[:, 0, :]
                    logits_e = classifier(cls_e)
                    epoch_val_loss += criterion(logits_e, lbl).item()
            epoch_val_loss /= len(val_dl)
            print(f"Epoch {epoch+1} validation loss: {epoch_val_loss:.4f}")

            thresh = 0.01 if best_val_loss < 0.1 else 0.1
            if best_val_loss - epoch_val_loss >= thresh:
                best_val_loss = epoch_val_loss
                path = os.path.join(ckpt_dir, f"best_epoch{epoch}_val_{epoch_val_loss:.4f}.pt")
                torch.save({
                    'epoch': epoch,
                    'iter': global_iter,
                    'model': model.state_dict(),
                    'clf': classifier.state_dict(),
                    'opt': optimizer.state_dict(),
                    'best_val_loss': best_val_loss
                }, path)
                print(f"Saved improved epoch checkpoint: {path}")

    ### Final Training Visualizations and Evaluation
    # Plot the full loss curve (or up to stopping point)
    plt.figure()
    plt.plot(iter_indices, iter_train_losses, label='Train Loss')
    if val_iter_indices:
        plt.plot(val_iter_indices, iter_val_losses, label='Val Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Full Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(ckpt_dir, 'loss_curve_full.png'))
    plt.show() # Display the final plot

    # Final evaluation on validation set
    all_labels, all_probs = [], []
    model.eval()
    classifier.eval() # Ensure classifier is also in eval mode
    with torch.no_grad():
        for batch in val_dl:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            lbl = batch['label'].to(device)
            out = model(input_ids=ids, attention_mask=mask)
            cls_e = out.last_hidden_state[:, 0, :]
            logits_e = classifier(cls_e)
            probs = torch.sigmoid(logits_e).squeeze().cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(lbl.cpu().tolist())

    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    cm = confusion_matrix(all_labels, preds)
    # plot confusion matrix
    plot_confusion_matrix(cm)
    # plot ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()


    
class AraTransformerDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }

def train_arabert(model, tokenizer, train_df, val_df, lr, weight_decay, epochs, checkpoint_dir="arabert_checkpoints", loss_check=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Print model type
    print(f"\n🔧 Training model: {model.__class__.__name__}\n")

    # Clean NaNs and convert to string
    train_df = train_df.dropna(subset=["stemmed_text", "source"])
    val_df = val_df.dropna(subset=["stemmed_text", "source"])

    train_texts = train_df["stemmed_text"].astype(str).tolist()
    val_texts = val_df["stemmed_text"].astype(str).tolist()

    # Tokenization
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")

    train_labels = torch.tensor(train_df["source"].values).float().unsqueeze(1)
    val_labels = torch.tensor(val_df["source"].values).float().unsqueeze(1)

    train_dataset = AraTransformerDataset(train_encodings, train_labels)
    val_dataset = AraTransformerDataset(val_encodings, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    loss_fn = nn.BCEWithLogitsLoss()

    os.makedirs(checkpoint_dir, exist_ok=True)

    model.train()
    losses, accs = [], []

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        print(f"\nEpoch {epoch+1}/{epochs}")
        loop = tqdm(train_loader, desc="Training", leave=False)

        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            logits = outputs.logits

            loss = loss_fn(logits, batch['labels'])
            total_loss += loss.item()

            preds = torch.round(torch.sigmoid(logits))
            correct += (preds == batch['labels']).sum().item()
            total += batch['labels'].size(0)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            loop.set_postfix(loss=loss.item())

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total
        losses.append(epoch_loss)
        accs.append(epoch_acc)

        if loss_check:
            print(f"✅ Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"arabert_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)

    return model, losses, accs


def evaluate_arabert(model, tokenizer, test_df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    test_encodings = tokenizer(test_df["stemmed_text"].tolist(), truncation=True, padding=True, max_length=512, return_tensors="pt")
    test_dataset = AraTransformerDataset(test_encodings, torch.zeros(len(test_df)))  # Dummy labels

    test_loader = DataLoader(test_dataset, batch_size=16)
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.sigmoid(logits).squeeze().cpu().tolist()
            all_preds.extend(probs)

    return all_preds

def train_araelectra(model: nn.Module,
                     tokenizer,
                     train_df,
                     val_df,
                     lr: float,
                     weight_decay: float,
                     epochs: int,
                     loss_check: int,
                     checkpoint_dir: str = None):

    model_type = model.config.model_type if hasattr(model.config, 'model_type') else 'araelectra'
    ckpt_dir = checkpoint_dir or f"checkpoints_{model_type}"
    prepare_ckpt_dir(ckpt_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    train_ds = TextDataset(train_df, tokenizer)
    val_ds = TextDataset(val_df, tokenizer)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    global_iter = 0
    best_val_loss = float('inf')

    iter_train_losses = []
    iter_val_losses = []
    iter_indices = []
    val_iter_indices = []

    stop_training = False

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}")
        epoch_train_loss = 0.0

        for batch in pbar:
            if keyboard and keyboard.is_pressed('q'): 
                stop_training = True

            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=masks)
            logits = outputs.logits
            loss = criterion(logits, labels.float())
            loss.backward()
            optimizer.step()

            global_iter += 1
            epoch_train_loss += loss.item()

            iter_indices.append(global_iter)
            iter_train_losses.append(loss.item())

            pbar.set_postfix({'loss': loss.item()})

            if global_iter % NUM_ITERS == 0:
                path = os.path.join(ckpt_dir, f"ckpt_iter{global_iter}.pt")
                torch.save({
                    'epoch': epoch,
                    'iter': global_iter,
                    'model': model.state_dict(),
                    'opt': optimizer.state_dict(),
                    'best_val_loss': best_val_loss
                }, path)
                print(f"\nSaved checkpoint at iter {global_iter}: {path}")
                plt.figure()
                plt.plot(iter_indices, iter_train_losses, label='Train Loss')
                if val_iter_indices:
                    plt.plot(val_iter_indices, iter_val_losses, label='Val Loss')
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.title(f'Loss Curve up to Iter {global_iter}')
                plt.legend()
                plt.savefig(os.path.join(ckpt_dir, f'loss_up_to_iter{global_iter}.png'))
                plt.close()

            if global_iter % loss_check == 0:
                model.eval()
                val_loss = 0.0
                sampled = random.sample(list(val_dl), max(1, len(val_dl)//100))
                with torch.no_grad():
                    for vb in sampled:
                        vi = vb['input_ids'].to(device)
                        vm = vb['attention_mask'].to(device)
                        vl = vb['label'].to(device)
                        out = model(input_ids=vi, attention_mask=vm)
                        logits_v = out.logits
                        val_loss += criterion(logits_v, vl.float()).item()
                val_loss /= len(sampled)

                avg_train_loss = epoch_train_loss / loss_check
                pbar.set_postfix({'loss': loss.item(), 'avg_train_loss': avg_train_loss, 'val_loss': val_loss})
                epoch_train_loss = 0.0

                iter_val_losses.append(val_loss)
                val_iter_indices.append(global_iter)

                thresh = 0.01 if best_val_loss < 0.1 else 0.1
                if best_val_loss - val_loss >= thresh:
                    best_val_loss = val_loss
                    path = os.path.join(ckpt_dir, f"best_iter{global_iter}_val_{val_loss:.4f}.pt")
                    torch.save({
                        'epoch': epoch,
                        'iter': global_iter,
                        'model': model.state_dict(),
                        'opt': optimizer.state_dict(),
                        'best_val_loss': best_val_loss
                    }, path)
                    print(f"\nSaved improved checkpoint: {path}")

                model.train()

            if stop_training:
                print("\nStopping training by user request...")
                break

        if stop_training:
            plt.figure()
            plt.plot(iter_indices, iter_train_losses, label='Train Loss')
            if val_iter_indices:
                plt.plot(val_iter_indices, iter_val_losses, label='Val Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title(f'Loss Curve up to Iter {global_iter} (Stopped)')
            plt.legend()
            plt.savefig(os.path.join(ckpt_dir, f'loss_curve_stopped_iter{global_iter}.png'))
            plt.close()
            print(f"Saved loss curve at stop: {os.path.join(ckpt_dir, f'loss_curve_stopped_iter{global_iter}.png')}")

            path = os.path.join(ckpt_dir, f'last_iter{global_iter}.pt')
            torch.save({
                'epoch': epoch,
                'iter': global_iter,
                'model': model.state_dict(),
                'opt': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, path)
            print(f"Stopped by user. Saved last checkpoint: {path}")
            break

        if not stop_training:
            model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for batch in val_dl:
                    ids = batch['input_ids'].to(device)
                    mask = batch['attention_mask'].to(device)
                    lbl = batch['label'].to(device)
                    out = model(input_ids=ids, attention_mask=mask)
                    logits_e = out.logits
                    epoch_val_loss += criterion(logits_e, lbl.float()).item()
            epoch_val_loss /= len(val_dl)
            print(f"Epoch {epoch+1} validation loss: {epoch_val_loss:.4f}")

            thresh = 0.01 if best_val_loss < 0.1 else 0.1
            if best_val_loss - epoch_val_loss >= thresh:
                best_val_loss = epoch_val_loss
                path = os.path.join(ckpt_dir, f"best_epoch{epoch}_val_{epoch_val_loss:.4f}.pt")
                torch.save({
                    'epoch': epoch,
                    'iter': global_iter,
                    'model': model.state_dict(),
                    'opt': optimizer.state_dict(),
                    'best_val_loss': best_val_loss
                }, path)
                print(f"Saved improved epoch checkpoint: {path}")

    plt.figure()
    plt.plot(iter_indices, iter_train_losses, label='Train Loss')
    if val_iter_indices:
        plt.plot(val_iter_indices, iter_val_losses, label='Val Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Full Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(ckpt_dir, 'loss_curve_full.png'))
    plt.show()

    all_labels, all_probs = [], []
    model.eval()
    with torch.no_grad():
        for batch in val_dl:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            lbl = batch['label'].to(device)
            out = model(input_ids=ids, attention_mask=mask)
            logits_e = out.logits
            probs = torch.sigmoid(logits_e).squeeze().cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(lbl.cpu().tolist())

    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    cm = confusion_matrix(all_labels, preds)
    plot_confusion_matrix(cm)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()
