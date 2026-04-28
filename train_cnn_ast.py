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
import shutil
from sklearn.metrics import confusion_matrix, recall_score, classification_report

from src.dataset import ASTDataset
from src.model_cnn_ast import CustomAST_CNN
from src.sam import SAM


# ─────────────────────────────────────────────────────────────────────────────
# FOCAL LOSS
#
# Pourquoi Focal Loss au lieu de CrossEntropy ?
#   - CrossEntropy traite tous les exemples pareil
#   - Focal Loss pénalise davantage les exemples DIFFICILES (Both, Wheeze)
#   - gamma=2 : les exemples bien classés ont un poids réduit de 75%
#   - Les class_weights amplifient encore plus les classes rares
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight   # class weights pour renforcer les classes rares

    def forward(self, inputs, targets):
        # Cross-entropy de base (sans réduction pour appliquer gamma)
        ce    = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt    = torch.exp(-ce)                         # probabilité de la bonne classe
        focal = ((1 - pt) ** self.gamma) * ce          # poids adaptatif
        return focal.mean()


def train(args):

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️  Device: {DEVICE}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── Chargement des données ────────────────────────────────────────────────
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

    label_names  = ['Normal', 'Crackle', 'Wheeze', 'Both']
    counts_train = np.bincount(y_train)

    print("\n📊 Distribution des classes (train) :")
    for i, name in enumerate(label_names):
        pct = counts_train[i] / len(y_train) * 100
        print(f"   {name:10s} : {counts_train[i]:4d} ({pct:.1f}%)")
    print(f"   {'Total':10s} : {len(y_train)}")

    # ── Class weights (inversement proportionnels à la fréquence) ─────────────
    # Pourquoi : Both et Wheeze sont très rares → on leur donne plus de poids
    # pour que le modèle les apprenne autant que Normal et Crackle
    class_weights        = 1.0 / counts_train.astype(np.float32)
    class_weights        = class_weights / class_weights.sum() * len(label_names)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    print("\n⚖️  Class weights :")
    for i, name in enumerate(label_names):
        print(f"   {name:10s} : {class_weights[i]:.4f}")

    # ── Processor AST ─────────────────────────────────────────────────────────
    processor = ASTFeatureExtractor.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    )

    # ── WeightedRandomSampler ─────────────────────────────────────────────────
    # Garantit un tirage équilibré des classes à chaque batch
    sampler_weights = [1.0 / counts_train[y] for y in y_train]
    sampler         = WeightedRandomSampler(sampler_weights, len(y_train))

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(
        ASTDataset(X_train, y_train, d_train, processor, train=True),
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )
    test_loader = DataLoader(
        ASTDataset(X_test, y_test, d_test, processor, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # ── Modèle CNN + AST ──────────────────────────────────────────────────────
    print("\n🧠 Preparing CNN + AST model...")
    model = CustomAST_CNN(num_classes=4, unfreeze_last_n=args.unfreeze_layers).to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"   Paramètres entraînables : {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
    print(f"   Couches AST dégelées    : {args.unfreeze_layers} dernières")

    # ── Optimiseur SAM ────────────────────────────────────────────────────────
    base_optimizer = torch.optim.AdamW
    optimizer      = SAM(
        model.parameters(), base_optimizer,
        lr=args.lr, rho=0.05, weight_decay=1e-4
    )

    # ── Focal Loss + class weights ────────────────────────────────────────────
    # AMÉLIORATION PRINCIPALE vs version précédente :
    #   Avant : CrossEntropyLoss(label_smoothing=0.1)
    #   Après : FocalLoss(gamma=2) + class weights calculés sur les données réelles
    criterion = FocalLoss(gamma=args.gamma, weight=class_weights_tensor)
    print(f"🎯 Loss : FocalLoss(gamma={args.gamma}) + class weights ✓")

    # ── OneCycleLR Scheduler ──────────────────────────────────────────────────
    # Pourquoi : LR fixe à 1e-5 ralentit la convergence sur les Transformer
    # OneCycleLR fait : warmup rapide → peak → descente cosine
    # Résultat : convergence plus rapide et meilleur minimum
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer.base_optimizer,
        max_lr=args.lr * 10,           # pic à 10× le LR initial = 1e-4
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.1,                 # 10% du training = warmup
        anneal_strategy='cos'
    )
    print(f"📈 Scheduler : OneCycleLR (max_lr={args.lr * 10:.0e}, warmup=10%) ✓")

    # ── Reprise depuis checkpoint ─────────────────────────────────────────────
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
        if 'scheduler_state' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch       = ckpt['epoch']
        best_score        = ckpt['best_score']
        best_recall_macro = ckpt['best_recall_macro']
        history           = ckpt['history']
        print(f"   ✅ Reprise epoch {start_epoch + 1}/{args.epochs}")
        print(f"   ✅ Meilleur recall macro : {best_recall_macro:.4f}")
    else:
        print("\n🆕 Démarrage depuis l'epoch 1")

    # ── Boucle d'entraînement ─────────────────────────────────────────────────
    print("🚀 Train begins\n")

    for epoch in range(start_epoch, args.epochs):

        # ── Dégel progressif : à partir de l'epoch 15, on dégèle toutes les couches
        if epoch == 14 and args.unfreeze_layers < 12:
            print(f"\n🔓 Epoch {epoch+1} : dégel complet de toutes les couches AST")
            for param in model.ast.encoder.parameters():
                param.requires_grad = True
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"   Paramètres entraînables : {trainable:,} / {total:,} ({trainable/total*100:.1f}%)\n")

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

            # Scheduler step (après chaque batch, pas epoch)
            scheduler.step()

            running_loss += loss.item()
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}', 'LR': f'{current_lr:.2e}'})

        # ── Évaluation ────────────────────────────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels, _ in test_loader:
                inputs = inputs.to(DEVICE)
                preds  = torch.argmax(model(inputs), dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        # ── Métriques ─────────────────────────────────────────────────────────
        cm    = confusion_matrix(all_labels, all_preds)
        se    = (np.sum(cm[1:, 1:]) / np.sum(cm[1:, :])) if np.sum(cm[1:, :]) > 0 else 0
        sp    = (cm[0, 0] / np.sum(cm[0, :])) if np.sum(cm[0, :]) > 0 else 0
        score = (se + sp) / 2

        recall_macro     = recall_score(all_labels, all_preds, average='macro',  zero_division=0)
        recall_per_class = recall_score(all_labels, all_preds, average=None,     zero_division=0, labels=[0,1,2,3])
        avg_loss         = running_loss / len(train_loader)
        current_lr       = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch+1:02d}/{args.epochs} | "
              f"Loss={avg_loss:.4f} | LR={current_lr:.2e} | "
              f"Score={score:.4f} (Se={se:.2f}, Sp={sp:.2f}) | "
              f"Recall macro={recall_macro:.4f}")
        print(f"         Recall → Normal:{recall_per_class[0]:.3f} "
              f"Crackle:{recall_per_class[1]:.3f} "
              f"Wheeze:{recall_per_class[2]:.3f} "
              f"Both:{recall_per_class[3]:.3f}")

        history.append({
            'epoch'         : epoch + 1,
            'loss'          : round(avg_loss, 4),
            'lr'            : round(current_lr, 8),
            'score'         : round(score, 4),
            'se'            : round(se, 4),
            'sp'            : round(sp, 4),
            'recall_macro'  : round(recall_macro, 4),
            'recall_normal' : round(float(recall_per_class[0]), 4),
            'recall_crackle': round(float(recall_per_class[1]), 4),
            'recall_wheeze' : round(float(recall_per_class[2]), 4),
            'recall_both'   : round(float(recall_per_class[3]), 4),
        })

        # ── Sauvegarde meilleur modèle ────────────────────────────────────────
        if recall_macro > best_recall_macro:
            best_recall_macro = recall_macro
            best_score        = score
            torch.save(model.state_dict(),
                       os.path.join(args.checkpoint_dir, "best_model_cnn_ast.pth"))
            print(f"   --> 💾 Best recall macro saved : {best_recall_macro:.4f}")

        # ── Resume checkpoint ─────────────────────────────────────────────────
        torch.save({
            'epoch'             : epoch + 1,
            'model_state'       : model.state_dict(),
            'optimizer_state'   : optimizer.state_dict(),
            'scheduler_state'   : scheduler.state_dict(),
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
    print(classification_report(all_labels, all_preds,
                                target_names=label_names,
                                digits=4, zero_division=0))

    results = {
        'method'            : f'CNN_AST_FocalLoss_gamma{args.gamma}_SAM_OneCycleLR',
        'hyperparameters'   : {
            'epochs'          : args.epochs,
            'batch_size'      : args.batch_size,
            'lr'              : args.lr,
            'gamma'           : args.gamma,
            'unfreeze_layers' : args.unfreeze_layers,
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
        description="CNN + AST + FocalLoss + OneCycleLR + SAM pour ICBHI"
    )
    parser.add_argument("--data_path",        type=str,   default="./icbhi_ast_16k_8s_metadata.npz")
    parser.add_argument("--checkpoint_dir",   type=str,   default="./checkpoints")
    parser.add_argument("--epochs",           type=int,   default=30)
    parser.add_argument("--batch_size",       type=int,   default=8)
    parser.add_argument("--lr",               type=float, default=1e-5)
    parser.add_argument("--gamma",            type=float, default=2.0,
                        help="Focal Loss gamma. 0=CrossEntropy, 2=recommandé")
    parser.add_argument("--unfreeze_layers",  type=int,   default=6,
                        help="Nombre de dernières couches AST à dégeler (défaut: 6)")
    parser.add_argument("--restart",          action="store_true",
                        help="Ignorer le checkpoint et repartir de zéro")

    args = parser.parse_args()
    train(args)
