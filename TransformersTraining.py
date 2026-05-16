import os
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_curve
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import RobertaConfig, RobertaModel

try:
    import keyboard
except ImportError:
    keyboard = None


BATCH_SIZE = 16
MAX_LEN = 256
NUM_ITERS = 1000


def prepare_ckpt_dir(path: str):
    os.makedirs(path, exist_ok=True)


class TextDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df["cleaned_text"].astype(str).tolist()
        self.labels = df["source"].astype(float).tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.float).unsqueeze(0),
        }


def build_roberta(config_name: str = "roberta-base") -> RobertaModel:
    config = RobertaConfig.from_pretrained(config_name)
    return RobertaModel(config)


def plot_confusion_matrix(cm):
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ["0", "1"])
    plt.yticks(tick_marks, ["0", "1"])
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()


def train_Transformers(
    model: nn.Module,
    tokenizer,
    train_df,
    val_df,
    lr: float,
    weight_decay: float,
    epochs: int,
    loss_check: int,
    checkpoint_dir: str = None,
    class_weights=None,
):
    mtype = model.config.model_type if hasattr(model.config, "model_type") else "roberta"
    ckpt_dir = checkpoint_dir or f"checkpoints_{mtype}"
    prepare_ckpt_dir(ckpt_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden_size = model.config.hidden_size
    classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(hidden_size, 1),
    ).to(device)
    model = model.to(device)

    train_ds = TextDataset(train_df, tokenizer)
    val_ds = TextDataset(val_df, tokenizer)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

    optimizer = AdamW(
        list(model.parameters()) + list(classifier.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    # For the binary single-logit classifier, balanced class weights map to BCE pos_weight.
    if class_weights is not None:
        class_weights = torch.as_tensor(class_weights, dtype=torch.float, device=device).flatten()
        if class_weights.numel() != 2:
            raise ValueError("Transformer weighted binary loss expects exactly two class weights.")
        pos_weight = (class_weights[1] / class_weights[0]).reshape(1)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print({"transformer_pos_weight": round(float(pos_weight.item()), 4)})
    else:
        criterion = nn.BCEWithLogitsLoss()

    global_iter = 0
    best_val_loss = float("inf")

    iter_train_losses = []
    iter_val_losses = []
    iter_indices = []
    val_iter_indices = []

    stop_training = False

    for epoch in range(epochs):
        model.train()
        classifier.train()
        pbar = tqdm(train_dl, desc=f"Epoch {epoch + 1}")
        epoch_train_loss = 0.0

        for batch in pbar:
            if keyboard is not None and keyboard.is_pressed("q"):
                stop_training = True

            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            masks = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

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
            pbar.set_postfix({"loss": loss.item()})

            if global_iter % NUM_ITERS == 0:
                path = os.path.join(ckpt_dir, f"ckpt_iter{global_iter}.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "iter": global_iter,
                        "model": model.state_dict(),
                        "clf": classifier.state_dict(),
                        "opt": optimizer.state_dict(),
                        "best_val_loss": best_val_loss,
                    },
                    path,
                )
                print(f"\nSaved checkpoint at iter {global_iter}: {path}")
                plt.figure()
                plt.plot(iter_indices, iter_train_losses, label="Train Loss")
                if val_iter_indices:
                    plt.plot(val_iter_indices, iter_val_losses, label="Val Loss")
                plt.xlabel("Iteration")
                plt.ylabel("Loss")
                plt.title(f"Loss Curve up to Iter {global_iter}")
                plt.legend()
                plt.savefig(os.path.join(ckpt_dir, f"loss_up_to_iter{global_iter}.png"))
                plt.close()

            if global_iter % loss_check == 0:
                model.eval()
                classifier.eval()
                val_loss = 0.0
                sampled = random.sample(list(val_dl), max(1, len(val_dl) // 100))
                with torch.no_grad():
                    for vb in sampled:
                        vi = vb["input_ids"].to(device)
                        vm = vb["attention_mask"].to(device)
                        vl = vb["label"].to(device)
                        out = model(input_ids=vi, attention_mask=vm)
                        cls_v = out.last_hidden_state[:, 0, :]
                        logits_v = classifier(cls_v)
                        val_loss += criterion(logits_v, vl.float()).item()
                val_loss /= len(sampled)

                avg_train_loss = epoch_train_loss / loss_check
                pbar.set_postfix(
                    {
                        "loss": loss.item(),
                        "avg_train_loss": avg_train_loss,
                        "val_loss": val_loss,
                    }
                )
                epoch_train_loss = 0.0

                iter_val_losses.append(val_loss)
                val_iter_indices.append(global_iter)

                thresh = 0.01 if best_val_loss < 0.1 else 0.1
                if best_val_loss - val_loss >= thresh:
                    best_val_loss = val_loss
                    path = os.path.join(ckpt_dir, f"best_iter{global_iter}_val_{val_loss:.4f}.pt")
                    torch.save(
                        {
                            "epoch": epoch,
                            "iter": global_iter,
                            "model": model.state_dict(),
                            "clf": classifier.state_dict(),
                            "opt": optimizer.state_dict(),
                            "best_val_loss": best_val_loss,
                        },
                        path,
                    )
                    print(f"\nSaved improved checkpoint: {path}")

                model.train()
                classifier.train()

            if stop_training:
                print("\nStopping training by user request...")
                break

        if stop_training:
            plt.figure()
            plt.plot(iter_indices, iter_train_losses, label="Train Loss")
            if val_iter_indices:
                plt.plot(val_iter_indices, iter_val_losses, label="Val Loss")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.title(f"Loss Curve up to Iter {global_iter} (Stopped)")
            plt.legend()
            plt.savefig(os.path.join(ckpt_dir, f"loss_curve_stopped_iter{global_iter}.png"))
            plt.close()
            print(f"Saved loss curve at stop: {os.path.join(ckpt_dir, f'loss_curve_stopped_iter{global_iter}.png')}")

            path = os.path.join(ckpt_dir, f"last_iter{global_iter}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "iter": global_iter,
                    "model": model.state_dict(),
                    "clf": classifier.state_dict(),
                    "opt": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                path,
            )
            print(f"Stopped by user. Saved last checkpoint: {path}")
            break

        model.eval()
        classifier.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                lbl = batch["label"].to(device)
                out = model(input_ids=ids, attention_mask=mask)
                cls_e = out.last_hidden_state[:, 0, :]
                logits_e = classifier(cls_e)
                epoch_val_loss += criterion(logits_e, lbl).item()
        epoch_val_loss /= len(val_dl)
        print(f"Epoch {epoch + 1} validation loss: {epoch_val_loss:.4f}")

        thresh = 0.01 if best_val_loss < 0.1 else 0.1
        if best_val_loss - epoch_val_loss >= thresh:
            best_val_loss = epoch_val_loss
            path = os.path.join(ckpt_dir, f"best_epoch{epoch}_val_{epoch_val_loss:.4f}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "iter": global_iter,
                    "model": model.state_dict(),
                    "clf": classifier.state_dict(),
                    "opt": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                path,
            )
            print(f"Saved improved epoch checkpoint: {path}")

    plt.figure()
    plt.plot(iter_indices, iter_train_losses, label="Train Loss")
    if val_iter_indices:
        plt.plot(val_iter_indices, iter_val_losses, label="Val Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Full Loss Curve")
    plt.legend()
    plt.savefig(os.path.join(ckpt_dir, "loss_curve_full.png"))
    plt.show()

    all_labels, all_probs = [], []
    model.eval()
    classifier.eval()
    with torch.no_grad():
        for batch in val_dl:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbl = batch["label"].to(device)
            out = model(input_ids=ids, attention_mask=mask)
            cls_e = out.last_hidden_state[:, 0, :]
            logits_e = classifier(cls_e)
            probs = torch.sigmoid(logits_e).view(-1).cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(lbl.view(-1).cpu().tolist())

    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    all_labels_int = [int(label) for label in all_labels]
    cm = confusion_matrix(all_labels_int, preds, labels=[0, 1])
    print(f"Validation Accuracy:  {accuracy_score(all_labels_int, preds):.4f}")
    print(f"Validation Precision: {precision_score(all_labels_int, preds, zero_division=0):.4f}")
    print(f"Validation Recall:    {recall_score(all_labels_int, preds, zero_division=0):.4f}")
    print(f"Validation F1 Score:  {f1_score(all_labels_int, preds, zero_division=0):.4f}")
    print("Classification report:")
    print(classification_report(all_labels_int, preds, labels=[0, 1], target_names=["human", "ai"], zero_division=0))
    plot_confusion_matrix(cm)
    fpr, tpr, _ = roc_curve(all_labels_int, all_probs)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.show()
