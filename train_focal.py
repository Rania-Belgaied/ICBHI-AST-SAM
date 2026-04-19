import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from transformers import ASTFeatureExtractor
from tqdm import tqdm
import os
import argparse
import json
from sklearn.metrics import confusion_matrix, recall_score, classification_report

from src.dataset import ASTDataset
from src.model import CustomAST
from src.sam import SAM


# ─────────────────────────────────────────────────────────────────────────────
# FOCAL LOSS
# Pourquoi : CrossEntropyLoss traite tous les exemples de façon égale.
# Le dataset ICBHI est très déséquilibré (Normal ~53%), donc le modèle apprend
# à prédire "Normal" par défaut et ignore les classes rares.
# Focal Loss ajoute un facteur (1 - pt)^gamma qui réduit la contribution des
# exemples faciles (Normal bien prédit) et force le modèle à apprendre les
# exemples difficiles (Crackle, Wheeze, Both).
# gamma=0  → identique à CrossEntropy classique
# gamma=2  → configuration recommandée pour données médicales déséquilibrées
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight  # class weights pour pénaliser encore plus les classes rares

    def forward(self, inputs, targets):
        # Cross-entropy par exemple (sans réduction pour pouvoir appliquer le facteur focal)
        ce = F.cross_entropy(inputs, targets,
                             weight=self.weight,
                             reduction='none')
        # pt = probabilité assignée à la vraie classe
        # Si le modèle est très confiant et a raison → pt proche de 1 → facteur proche de 0
        # Si le modèle se trompe ou est incertain → pt proche de 0 → facteur proche de 1
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


def train(args):

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️  Device: {DEVICE}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── Chargement des données ────────────────────────────────────────────────
    print(f"📥 Loading: {args.data_path}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(
            f"Fichier introuvable : {args.data_path}. Lance preprocess.py d'abord."
        )

    data    = np.load(args.data_path)
    X_train = data['X_train']
    y_train = data['y_train']
    d_train = data['device_train']
    X_test  = data['X_test']
    y_test  = data['y_test']
    d_test  = data['device_test']

    # ── Afficher la distribution réelle des classes ───────────────────────────
    # Pourquoi : confirmer le déséquilibre avant de calculer les poids
    label_names = ['Normal', 'Crackle', 'Wheeze', 'Both']
    counts_train = np.bincount(y_train)
    print("\n📊 Distribution des classes (train) :")
    for i, name in enumerate(label_names):
        pct = counts_train[i] / len(y_train) * 100
        print(f"   {name:10s} : {counts_train[i]:4d} ({pct:.1f}%)")
    print(f"   {'Total':10s} : {len(y_train)}")

    # ── Class weights ─────────────────────────────────────────────────────────
    # Pourquoi : donner plus de poids aux classes rares dans la loss.
    # Poids = 1 / fréquence, puis normalisé pour sommer à 1.
    # Résultat attendu : Both aura le poids le plus fort, Normal le plus faible.
    class_weights = 1.0 / counts_train.astype(np.float32)
    class_weights = class_weights / class_weights.sum()
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    print("\n⚖️  Class weights calculés :")
    for i, name in enumerate(label_names):
        print(f"   {name:10s} : {class_weights[i]:.4f}")

    # ── Processor AST ─────────────────────────────────────────────────────────
    processor = ASTFeatureExtractor.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    )

    # ── Sampler (identique à train.py) ───────────────────────────────────────
    # Pourquoi : le WeightedRandomSampler rééquilibre déjà les batches au niveau
    # de l'échantillonnage. Combiné à la Focal Loss, l'effet est double.
    sampler_weights = [1.0 / counts_train[y] for y in y_train]
    sampler = WeightedRandomSampler(sampler_weights, len(y_train))

    # ── DataLoaders ───────────────────────────────────────────────────────────
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

    # ── Modèle et optimiseur (identiques à train.py) ──────────────────────────
    # Pourquoi garder SAM : on veut isoler l'effet de la Focal Loss.
    # La seule variable qui change par rapport à train.py est le criterion.
    print("\n🧠 Preparing model...")
    model = CustomAST(num_classes=4).to(DEVICE)

    base_optimizer = torch.optim.AdamW
    optimizer = SAM(
        model.parameters(), base_optimizer,
        lr=args.lr, rho=0.05, weight_decay=1e-4
    )

    # ── Criterion : Focal Loss + class weights ────────────────────────────────
    # C'est la seule modification par rapport à train.py original :
    # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  ← original
    # criterion = FocalLoss(gamma=2.0, weight=...)           ← notre version
    criterion = FocalLoss(gamma=args.gamma, weight=class_weights_tensor)
    print(f"🎯 Loss : FocalLoss(gamma={args.gamma}) + class weights ✓")

    # ── Boucle d'entraînement ─────────────────────────────────────────────────
    print("🚀 Train begins\n")
    best_score        = 0.0
    best_recall_macro = 0.0
    history           = []

    for epoch in range(args.epochs):
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
                logits = model(inputs)
                preds  = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        # ── Métriques ─────────────────────────────────────────────────────────
        # Score original de l'article (Se + Sp) / 2 — gardé pour comparaison
        cm = confusion_matrix(all_labels, all_preds)
        se = (np.sum(cm[1:, 1:]) / np.sum(cm[1:, :])) if np.sum(cm[1:, :]) > 0 else 0
        sp = (cm[0, 0] / np.sum(cm[0, :])) if np.sum(cm[0, :]) > 0 else 0
        score = (se + sp) / 2

        # Recall macro — métrique cible du projet
        recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)

        # Recall par classe
        recall_per_class = recall_score(
            all_labels, all_preds,
            average=None,
            zero_division=0,
            labels=[0, 1, 2, 3]
        )

        avg_loss = running_loss / len(train_loader)

        print(f"Epoch {epoch+1:02d}/{args.epochs} | "
              f"Loss={avg_loss:.4f} | "
              f"Score={score:.4f} (Se={se:.2f}, Sp={sp:.2f}) | "
              f"Recall macro={recall_macro:.4f}")
        print(f"         Recall → Normal:{recall_per_class[0]:.3f} "
              f"Crackle:{recall_per_class[1]:.3f} "
              f"Wheeze:{recall_per_class[2]:.3f} "
              f"Both:{recall_per_class[3]:.3f}")

        # Historique pour sauvegarde finale
        history.append({
            'epoch'        : epoch + 1,
            'loss'         : round(avg_loss, 4),
            'score'        : round(score, 4),
            'se'           : round(se, 4),
            'sp'           : round(sp, 4),
            'recall_macro' : round(recall_macro, 4),
            'recall_normal' : round(float(recall_per_class[0]), 4),
            'recall_crackle': round(float(recall_per_class[1]), 4),
            'recall_wheeze' : round(float(recall_per_class[2]), 4),
            'recall_both'   : round(float(recall_per_class[3]), 4),
        })

        # Sauvegarde du meilleur modèle selon le recall macro
        # Pourquoi recall_macro et non score : c'est la métrique cible du projet
        if recall_macro > best_recall_macro:
            best_recall_macro = recall_macro
            best_score        = score
            save_path = os.path.join(args.checkpoint_dir, "best_model_focal.pth")
            torch.save(model.state_dict(), save_path)
            print(f"   --> 💾 Best recall macro saved : {best_recall_macro:.4f}")

    # ── Résultats finaux ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"🏆 Best Score (Se+Sp)/2  : {best_score:.4f}")
    print(f"🎯 Best Recall macro     : {best_recall_macro:.4f}  ← métrique cible")
    print(f"{'='*60}")

    # Classification report complet sur la dernière epoch
    print("\n📋 Classification report (dernière epoch) :")
    print(classification_report(
        all_labels, all_preds,
        target_names=label_names,
        digits=4,
        zero_division=0
    ))

    # ── Sauvegarde JSON des résultats ─────────────────────────────────────────
    # Pourquoi : Kaggle ferme les sessions — les résultats doivent être persistés
    results = {
        'method'            : f'FocalLoss_gamma{args.gamma}_classweights',
        'hyperparameters'   : {
            'epochs'    : args.epochs,
            'batch_size': args.batch_size,
            'lr'        : args.lr,
            'gamma'     : args.gamma,
        },
        'best_recall_macro' : best_recall_macro,
        'best_score'        : best_score,
        'history'           : history,
        'last_epoch_report' : classification_report(
            all_labels, all_preds,
            target_names=label_names,
            output_dict=True,
            zero_division=0
        )
    }

    results_path = os.path.join(args.checkpoint_dir, "results_focal.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Résultats sauvegardés : {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train AST + SAM avec Focal Loss pour améliorer le recall (ICBHI)"
    )
    parser.add_argument("--data_path",       type=str,   default="./icbhi_ast_16k_8s_metadata.npz")
    parser.add_argument("--checkpoint_dir",  type=str,   default="./checkpoints")
    parser.add_argument("--epochs",          type=int,   default=20)
    parser.add_argument("--batch_size",      type=int,   default=8)
    parser.add_argument("--lr",              type=float, default=1e-5)
    parser.add_argument("--gamma",           type=float, default=2.0,
                        help="Focal Loss gamma. 0=CrossEntropy, 2=recommandé")

    args = parser.parse_args()
    train(args)