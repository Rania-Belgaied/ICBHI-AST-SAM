import torch
import torch.nn as nn
from transformers import ASTModel


# ─────────────────────────────────────────────────────────────────────────────
# CNN FEATURE EXTRACTOR
#
# Pourquoi 3 couches Conv2d :
#   - Couche 1 : détecte les contours et textures de bas niveau
#                (transitions brusques = crackles ~5ms)
#   - Couche 2 : combine les patterns pour former des structures
#                (oscillations répétées = wheezes ~400Hz)
#   - Couche 3 : features de haut niveau, prêtes pour le Transformer
#
# Pourquoi des petits filtres 3×3 :
#   - Capturent les patterns locaux dans le spectrogramme
#   - Légers en paramètres → pas d'overfitting sur ICBHI (petit dataset)
#
# Pourquoi BatchNorm + ReLU :
#   - BatchNorm stabilise l'entraînement (la loss ne diverge pas)
#   - ReLU introduit la non-linéarité nécessaire
# ─────────────────────────────────────────────────────────────────────────────
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            # Couche 1 : 1 canal → 32 feature maps
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),   # divise la résolution par 2

            # Couche 2 : 32 → 64 feature maps
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),   # divise la résolution par 2

            # Couche 3 : 64 → 128 feature maps
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Pas de MaxPool ici — on garde une résolution suffisante pour le Transformer
        )

    def forward(self, x):
        # x : (batch, freq, time) — sortie du ASTFeatureExtractor
        # On ajoute la dimension canal : (batch, 1, freq, time)
        x = x.unsqueeze(1)

        # Passage dans les 3 couches CNN
        # Sortie : (batch, 128, freq/4, time/4)
        features = self.cnn(x)

        return features


# ─────────────────────────────────────────────────────────────────────────────
# CNN + AST HYBRIDE
#
# Pipeline :
#   1. CNNFeatureExtractor  → extrait les patterns locaux du spectrogramme
#   2. Projection linéaire  → projette les features CNN vers 768 dims (espace AST)
#   3. ASTModel Transformer → analyse les relations globales entre features
#   4. Mean pooling         → agrège la séquence en un vecteur
#   5. Classifier           → prédit la classe (Normal/Crackle/Wheeze/Both)
# ─────────────────────────────────────────────────────────────────────────────
class CustomAST_CNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        # ── 1. CNN feature extractor ──────────────────────────────────────────
        self.cnn_extractor = CNNFeatureExtractor()

        # ── 2. Projection CNN → espace AST ───────────────────────────────────
        # Le CNN produit 128 feature maps.
        # L'AST attend des tokens de dimension 768.
        # On projette chaque position spatiale (token) de 128 → 768 dims.
        self.projection = nn.Sequential(
            nn.Linear(128, 768),
            nn.LayerNorm(768),    # stabilise l'entrée du Transformer
            nn.Dropout(0.1)
        )

        # ── 3. AST Transformer ────────────────────────────────────────────────
        # On charge l'AST pré-entraîné MAIS on ignore son patch embedding
        # car notre CNN le remplace.
        # On utilise uniquement les couches Transformer (self-attention).
        self.ast = ASTModel.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )

        # Geler les premières couches du Transformer (optionnel)
        # Pourquoi : les premières couches de l'AST ont appris des
        # représentations générales sur AudioSet — les garder gelées
        # évite de les écraser avec le petit dataset ICBHI.
        # On dégèle seulement les 4 dernières couches.
        total_layers = len(self.ast.encoder.layer)
        for i, layer in enumerate(self.ast.encoder.layer):
            if i < total_layers - 4:   # geler les N-4 premières couches
                for param in layer.parameters():
                    param.requires_grad = False

        # ── 4. Classifier ─────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768, num_classes)
        )

    def forward(self, x):
        # ── Étape 1 : extraction CNN ──────────────────────────────────────────
        # x : (batch, freq, time) — format sortie ASTFeatureExtractor
        cnn_features = self.cnn_extractor(x)
        # cnn_features : (batch, 128, freq/4, time/4)

        batch_size = cnn_features.shape[0]
        C = cnn_features.shape[1]   # 128 canaux

        # ── Étape 2 : reshape en séquence de tokens ───────────────────────────
        # Le Transformer attend : (batch, nb_tokens, dim)
        # On aplatit les dimensions spatiales (freq/4 × time/4) en tokens
        cnn_features = cnn_features.permute(0, 2, 3, 1)
        # (batch, freq/4, time/4, 128)

        cnn_features = cnn_features.reshape(batch_size, -1, C)
        # (batch, nb_tokens, 128)   où nb_tokens = freq/4 × time/4

        # ── Étape 3 : projection 128 → 768 ───────────────────────────────────
        tokens = self.projection(cnn_features)
        # (batch, nb_tokens, 768)

        # ── Étape 4 : Transformer (couches self-attention de l'AST) ──────────
        # On passe directement nos tokens aux couches encoder de l'AST
        # en contournant le patch embedding original
        encoder_output = self.ast.encoder(
            hidden_states=tokens,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )
        hidden_states = encoder_output.last_hidden_state
        # (batch, nb_tokens, 768)

        # LayerNorm finale (présente dans l'AST original)
        hidden_states = self.ast.layernorm(hidden_states)

        # ── Étape 5 : Mean pooling ────────────────────────────────────────────
        # Agrège tous les tokens en un seul vecteur représentatif
        embeddings = hidden_states.mean(dim=1)
        # (batch, 768)

        # ── Étape 6 : Classification ──────────────────────────────────────────
        logits = self.classifier(embeddings)
        # (batch, 4)

        return logits