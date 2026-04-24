import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from transformers import ASTFeatureExtractor
from tqdm import tqdm
import os
import argparse
import json
import shutil
from sklearn.metrics import confusion_matrix, recall_score, classification_report

from src.dataset import ASTDataset
from src.model_cnn_ast import CustomAST_CNN   # ← seul changement vs train.py
from src.sam import SAM


def train(args):

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️  Device: {DEVICE}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── Données ───────────────────────────────────────────────────────────────
    print(f"📥 Loading: {args.data_path}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Fichier introuvable : {args.data_path}")

    data    = np.load(args.data_path)
    X_train = data['X_train']
    y_train = data['y_train']
    d_train = data['device_train']
    X_test  = data['X_test']
    y_test  = data['y_test']
    d_test  = data['device_test']

    # ── Processor + Sampler + DataLoaders (identiques à train.py) ─────────────
    processor = ASTFeatureExtractor.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    )

    counts  = np.bincount(y_train)
    weights = [1.0 / counts[y] for y in y_train]
    sampler = WeightedRandomSampler(weights, len(y_train))

    train_loader = DataLoader(
        ASTDataset(X_train, y_train, d_train, processor, train=True),
        batch_size=args.batch_size,
        sampler=sampler
    )
    test_loader = DataLoader(
        ASTDataset(X_test, y_test, d_test, processor, train=False),
        batch_size=args.batch_size,
        shuffle=False
    )

    # ── Modèle CNN + AST ──────────────────────────────────────────────────────
    # Seule différence avec train.py :
    #   train.py       → CustomAST(num_classes=4)
    #   train_cnn_ast  → CustomAST_CNN(num_classes=4)
    # Tout le reste est identique à l'original.
    print("🧠 Preparing CNN + AST model...")
    model = CustomAST_CNN(num_classes=4).to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"   Paramètres entraînables : {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    # ── Optimiseur SAM + Loss (identiques à train.py) ─────────────────────────
    base_optimizer = torch.optim.AdamW
    optimizer      = SAM(
        model.parameters(), base_optimizer,
        lr=args.lr, rho=0.05, weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # identique à train.py

    # ── Resume checkpoint ─────────────────────────────────────────────────────
    resume_path       = os.path.join(args.checkpoint_dir, "resume_cnn_ast.pth")
    start_epoch       = 0
    best_score        = 0.0
    best_recall_macro = 0.0
    history           = []

    if os.path.exists(resume_path) and not args.restart:
        print(f"\n🔄 Reprise depuis : {resume_path}")
        ckpt = torch.load(resume_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch       = ckpt['epoch']
        best_score        = ckpt['best_score']
        best_recall_macro = ckpt['best_recall_macro']
        history           = ckpt['history']
        print(f"   ✅ Reprise epoch {start_epoch + 1}/{args.epochs}")
        print(f"   ✅ Meilleur recall macro : {best_recall_macro:.4f}")
    else:
        print("\n🆕 Démarrage depuis l'epoch 1")

    # ── Boucle d'entraînement (identique à train.py) ──────────────────────────
    print("🚀 Train begins\n")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            leave=False
        )

        for inputs, labels, _ in progress_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # SAM first step
            logits = model(inputs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # SAM second step
            criterion(model(inputs), labels).backward()
            optimizer.second_step(zero_grad=True)

            running_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        # ── Évaluation ────────────────────────────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels, _ in test_loader:
                inputs = inputs.to(DEVICE)
                preds  = torch.argmax(model(inputs), dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        # ── Métriques (score original + recall macro) ─────────────────────────
        cm    = confusion_matrix(all_labels, all_preds)
        se    = (np.sum(cm[1:, 1:]) / np.sum(cm[1:, :])) if np.sum(cm[1:, :]) > 0 else 0
        sp    = (cm[0, 0] / np.sum(cm[0, :])) if np.sum(cm[0, :]) > 0 else 0
        score = (se + sp) / 2

        recall_macro     = recall_score(all_labels, all_preds, average='macro',  zero_division=0)
        recall_per_class = recall_score(all_labels, all_preds, average=None,     zero_division=0, labels=[0,1,2,3])
        avg_loss         = running_loss / len(train_loader)

        print(f"Epoch {epoch+1:02d}/{args.epochs} | "
              f"Loss={avg_loss:.4f} | "
              f"Score={score:.4f} (Se={se:.2f}, Sp={sp:.2f}) | "
              f"Recall macro={recall_macro:.4f}")
        print(f"         Recall → Normal:{recall_per_class[0]:.3f} "
              f"Crackle:{recall_per_class[1]:.3f} "
              f"Wheeze:{recall_per_class[2]:.3f} "
              f"Both:{recall_per_class[3]:.3f}")

        history.append({
            'epoch'         : epoch + 1,
            'loss'          : round(avg_loss, 4),
            'score'         : round(score, 4),
            'se'            : round(se, 4),
            'sp'            : round(sp, 4),
            'recall_macro'  : round(recall_macro, 4),
            'recall_normal' : round(float(recall_per_class[0]), 4),
            'recall_crackle': round(float(recall_per_class[1]), 4),
            'recall_wheeze' : round(float(recall_per_class[2]), 4),
            'recall_both'   : round(float(recall_per_class[3]), 4),
        })

        # Sauvegarde meilleur modèle selon recall macro
        if recall_macro > best_recall_macro:
            best_recall_macro = recall_macro
            best_score        = score
            torch.save(model.state_dict(),
                       os.path.join(args.checkpoint_dir, "best_model_cnn_ast.pth"))
            print(f"   --> 💾 Best recall macro saved : {best_recall_macro:.4f}")

        # Resume checkpoint — écrasé à chaque epoch
        torch.save({
            'epoch'             : epoch + 1,
            'model_state'       : model.state_dict(),
            'optimizer_state'   : optimizer.state_dict(),
            'best_score'        : best_score,
            'best_recall_macro' : best_recall_macro,
            'history'           : history,
        }, resume_path)

        shutil.copy(resume_path, '/kaggle/working/resume_cnn_ast.pth')
        print(f"   --> 📌 Checkpoint sauvegardé (epoch {epoch+1}/{args.epochs})")

    # ── Résultats finaux ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"🏆 Best Score (Se+Sp)/2  : {best_score:.4f}")
    print(f"🎯 Best Recall macro     : {best_recall_macro:.4f}  ← métrique cible")
    print(f"{'='*60}")

    print("\n📋 Classification report (dernière epoch) :")
    label_names = ['Normal', 'Crackle', 'Wheeze', 'Both']
    print(classification_report(all_labels, all_preds,
                                target_names=label_names,
                                digits=4, zero_division=0))

    # Sauvegarde JSON
    results = {
        'method'            : 'CNN_AST_CrossEntropy_SAM',
        'hyperparameters'   : {
            'epochs'    : args.epochs,
            'batch_size': args.batch_size,
            'lr'        : args.lr,
        },
        'best_recall_macro' : best_recall_macro,
        'best_score'        : best_score,
        'history'           : history,
        'last_epoch_report' : classification_report(
            all_labels, all_preds,
            target_names=label_names,
            output_dict=True, zero_division=0
        )
    }

    results_path = os.path.join(args.checkpoint_dir, "results_cnn_ast.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    shutil.copy(results_path, '/kaggle/working/results_cnn_ast.json')
    shutil.copy(os.path.join(args.checkpoint_dir, 'best_model_cnn_ast.pth'),
                '/kaggle/working/best_model_cnn_ast.pth')

    print(f"\n✅ Résultats sauvegardés")
    print("📦 Fichiers copiés dans /kaggle/working/ ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CNN + AST avec SAM pour ICBHI — même config que train.py"
    )
    parser.add_argument("--data_path",      type=str,   default="./icbhi_ast_16k_8s_metadata.npz")
    parser.add_argument("--checkpoint_dir", type=str,   default="./checkpoints")
    parser.add_argument("--epochs",         type=int,   default=20)
    parser.add_argument("--batch_size",     type=int,   default=8)
    parser.add_argument("--lr",             type=float, default=1e-5)
    parser.add_argument("--restart",        action="store_true",
                        help="Ignorer le checkpoint et repartir de zéro")

    args = parser.parse_args()
    train(args)