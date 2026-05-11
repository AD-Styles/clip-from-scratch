"""
CLIP from Scratch — Contrastive Language-Image Pre-training in PyTorch
=======================================================================

CIFAR-10 + 합성 캡션으로 학습하는 미니 CLIP 구현.

핵심 구성 (이전 from-scratch 시리즈 재활용):
- Image Encoder: ResNet-20 (#26 resnet-from-scratch의 BasicBlock 재활용)
- Text Encoder: 4-layer Transformer Encoder (#25 transformer-from-scratch 재활용)
- Joint Space: 256-dim multimodal embedding
- Loss: Symmetric InfoNCE (이번 프로젝트의 핵심 학습 목표)

CLI:
    python main.py                # 학습 → 시각화
    python main.py --retrain      # 강제 재학습
    python main.py --viz-only     # 학습 건너뛰고 체크포인트로 시각화만
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# 한국어 폰트 설정 (matplotlib)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# ============================================================================
# Hyperparameters
# ============================================================================

# Architecture
EMBED_DIM = 256          # Joint multimodal space dim
TEXT_D_MODEL = 256       # Transformer hidden dim
TEXT_N_HEADS = 8         # Transformer heads
TEXT_N_LAYERS = 4        # Transformer encoder layers
TEXT_D_FF = 1024         # Transformer FFN dim
MAX_LEN = 16             # Max caption length (in tokens, including [SOS]/[EOS])

# Training
BATCH_SIZE = 256
EPOCHS = 30
LR = 5e-4
WEIGHT_DECAY = 0.2
DROPOUT = 0.1
INIT_TEMP = 0.07         # Initial temperature τ (CLIP 논문 기본값)

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CHECKPOINT_PATH = DATA_DIR / "checkpoint.pt"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INIT] Device: {DEVICE}, Epochs: {EPOCHS}, Batch: {BATCH_SIZE}", flush=True)


# ============================================================================
# Captions — CIFAR-10 + Synthetic Templates
# ============================================================================

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# 학습용 캡션 템플릿 — 클래스당 다양한 표현으로 학습
TRAIN_TEMPLATES = [
    "a photo of a {cls}",
    "an image of a {cls}",
    "a picture of a {cls}",
    "a small {cls}",
    "a big {cls}",
    "a clear photo of a {cls}",
    "a blurry {cls}",
    "a cropped image of a {cls}",
    "a low resolution {cls}",
    "this is a {cls}",
]

# Zero-shot 평가용 — 학습 때 한 번도 안 본 템플릿
EVAL_TEMPLATES = [
    "the {cls}",
    "a {cls} in the scene",
    "a colorful {cls}",
]


# ============================================================================
# Tokenizer — simple word-level
# ============================================================================

class Tokenizer:
    """간단한 단어 단위 토크나이저. [PAD]/[SOS]/[EOS]/[UNK] 특수 토큰 포함."""

    def __init__(self, all_texts: list[str]):
        # 학습 캡션에서 어휘 수집
        vocab = set()
        for text in all_texts:
            vocab.update(text.lower().split())

        # 특수 토큰
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3

        self.idx_to_word = ["[PAD]", "[SOS]", "[EOS]", "[UNK]"]
        self.word_to_idx = {w: i for i, w in enumerate(self.idx_to_word)}

        for word in sorted(vocab):
            self.word_to_idx[word] = len(self.idx_to_word)
            self.idx_to_word.append(word)

        self.vocab_size = len(self.idx_to_word)

    def encode(self, text: str, max_len: int = MAX_LEN) -> tuple[torch.Tensor, torch.Tensor]:
        """Text → (token_ids, mask) — mask는 1 for real, 0 for [PAD]."""
        words = text.lower().split()
        ids = [self.sos_idx] + [
            self.word_to_idx.get(w, self.unk_idx) for w in words
        ] + [self.eos_idx]
        ids = ids[:max_len]

        mask = [1] * len(ids)
        while len(ids) < max_len:
            ids.append(self.pad_idx)
            mask.append(0)

        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)

    def __len__(self):
        return self.vocab_size


def build_tokenizer() -> Tokenizer:
    """학습 + 평가 모든 캡션을 모아서 토크나이저 구축."""
    all_captions = []
    for tpl in TRAIN_TEMPLATES + EVAL_TEMPLATES:
        for cls in CIFAR10_CLASSES:
            all_captions.append(tpl.format(cls=cls))
    return Tokenizer(all_captions)


# ============================================================================
# Image Encoder — ResNet-20 (재활용: #26 resnet-from-scratch)
# ============================================================================

class BasicBlock(nn.Module):
    """ResNet 기본 블록 — He et al., CVPR 2016. #26 resnet-from-scratch에서 재활용.

    Skip connection: out = F(x) + shortcut(x)
    """

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)        # ★ 핵심: H(x) = F(x) + x
        return F.relu(out)


class ImageEncoder(nn.Module):
    """ResNet-20 (6n+2, n=3) + projection head → 256-dim image embedding.

    구조: Conv-BN-ReLU → Stage1(16→16) → Stage2(16→32) → Stage3(32→64) → GAP → Linear.
    마지막 FC를 256-dim projection head로 교체한 것 외에는 #26과 동일.
    """

    def __init__(self, n: int = 3, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.stage1 = self._make_stage(16, 16, n, stride=1)
        self.stage2 = self._make_stage(16, 32, n, stride=2)
        self.stage3 = self._make_stage(32, 64, n, stride=2)
        self.projection = nn.Linear(64, embed_dim, bias=False)

        # He 초기화 (논문 §3.4 / #26과 동일)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_stage(self, in_ch, out_ch, n_blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(n_blocks - 1):
            layers.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.projection(x)


# ============================================================================
# Text Encoder — Transformer (재활용: #25 transformer-from-scratch)
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding — #25 재활용.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    """

    def __init__(self, d_model: int, max_len: int = MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V — #25 재활용."""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, V)
    return out, attn


class MultiHeadAttention(nn.Module):
    """Multi-head attention — #25 재활용."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch = q.size(0)
        Q = self.W_q(q).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)  # head 차원 broadcast
        out, _ = scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(batch, -1, self.d_model)
        return self.dropout(self.W_o(out))


class PositionwiseFeedForward(nn.Module):
    """FFN — #25 재활용."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class EncoderLayer(nn.Module):
    """Encoder layer (Self-Attn + FFN with residual + LayerNorm) — #25 재활용."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.self_attn(x, x, x, mask))
        x = self.norm2(x + self.ff(x))
        return x


class TextEncoder(nn.Module):
    """Transformer Encoder + projection head → 256-dim text embedding.

    캡션 토큰 → Embedding + PE → N개 Encoder Layer → Mean Pool(PAD 제외) → Projection.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = TEXT_D_MODEL,
        n_heads: int = TEXT_N_HEADS,
        n_layers: int = TEXT_N_LAYERS,
        d_ff: int = TEXT_D_FF,
        max_len: int = MAX_LEN,
        embed_dim: int = EMBED_DIM,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.projection = nn.Linear(d_model, embed_dim, bias=False)
        self.d_model = d_model

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):
        """tokens: (B, L), mask: (B, L) — 1 for real, 0 for [PAD]."""
        x = self.token_embed(tokens) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        # Attention mask: padding key 차단. (B, L) → (B, 1, L) → broadcast.
        attn_mask = mask.unsqueeze(1)
        for layer in self.layers:
            x = layer(x, attn_mask)
        # Mean pooling over non-padded tokens
        mask_f = mask.unsqueeze(-1).float()
        x = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-6)
        return self.projection(x)


# ============================================================================
# CLIP — Image + Text + Symmetric InfoNCE
# ============================================================================

class CLIP(nn.Module):
    """CLIP — Image encoder + Text encoder + learnable temperature."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.image_encoder = ImageEncoder(n=3, embed_dim=EMBED_DIM)
        self.text_encoder = TextEncoder(vocab_size=vocab_size, embed_dim=EMBED_DIM)
        # Learnable temperature τ (CLIP 논문 ablation: log scale로 학습이 안정적)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / INIT_TEMP)))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        emb = self.image_encoder(images)
        return F.normalize(emb, dim=-1)

    def encode_text(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        emb = self.text_encoder(tokens, mask)
        return F.normalize(emb, dim=-1)

    def forward(self, images, tokens, mask):
        img_emb = self.encode_image(images)
        txt_emb = self.encode_text(tokens, mask)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits = logit_scale * img_emb @ txt_emb.T   # (B, B)
        return logits, img_emb, txt_emb


def info_nce_loss(logits: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE — image→text CE + text→image CE, 평균."""
    batch_size = logits.size(0)
    labels = torch.arange(batch_size, device=logits.device)
    loss_i = F.cross_entropy(logits, labels)       # 각 행: 자기 짝인 텍스트 찾기
    loss_t = F.cross_entropy(logits.T, labels)     # 각 열: 자기 짝인 이미지 찾기
    return (loss_i + loss_t) / 2


def retrieval_top1(logits: torch.Tensor) -> tuple[float, float]:
    """배치 내 top-1 retrieval 정확도 (image→text, text→image)."""
    batch_size = logits.size(0)
    labels = torch.arange(batch_size, device=logits.device)
    i2t = (logits.argmax(dim=-1) == labels).float().mean().item()
    t2i = (logits.argmax(dim=0) == labels).float().mean().item()
    return i2t, t2i


# ============================================================================
# Dataset — CIFAR-10 + Synthetic Captions
# ============================================================================

# CIFAR-10 표준 정규화 (#26과 동일)
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


class CIFAR10Captioned(Dataset):
    """CIFAR-10 + 샘플별 합성 캡션. 매 epoch마다 같은 이미지가 같은 캡션을 받도록 고정."""

    def __init__(self, cifar_dataset, templates: list[str], tokenizer: Tokenizer, seed: int = SEED):
        self.cifar = cifar_dataset
        self.templates = templates
        self.tokenizer = tokenizer
        # 샘플별로 고정된 템플릿 인덱스 (재현성 + 학습 안정성)
        rng = np.random.RandomState(seed)
        self.template_indices = rng.randint(0, len(templates), size=len(cifar_dataset))

    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, idx):
        image, label = self.cifar[idx]
        cls_name = CIFAR10_CLASSES[label]
        template = self.templates[self.template_indices[idx]]
        caption = template.format(cls=cls_name)
        tokens, mask = self.tokenizer.encode(caption)
        return image, tokens, mask, label, caption


def collate_fn(batch):
    """default collate가 string list를 잘 다루도록 직접 구성."""
    images = torch.stack([b[0] for b in batch])
    tokens = torch.stack([b[1] for b in batch])
    masks = torch.stack([b[2] for b in batch])
    labels = torch.tensor([b[3] for b in batch], dtype=torch.long)
    captions = [b[4] for b in batch]
    return images, tokens, masks, labels, captions


def get_dataloaders(tokenizer: Tokenizer):
    """CIFAR-10 + 합성 캡션 DataLoader 구성."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_cifar = torchvision.datasets.CIFAR10(
        str(DATA_DIR), train=True, download=True, transform=transform,
    )
    test_cifar = torchvision.datasets.CIFAR10(
        str(DATA_DIR), train=False, download=True, transform=transform,
    )
    train_set = CIFAR10Captioned(train_cifar, TRAIN_TEMPLATES, tokenizer, seed=SEED)
    test_set = CIFAR10Captioned(test_cifar, TRAIN_TEMPLATES, tokenizer, seed=SEED + 1)

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True, collate_fn=collate_fn, drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=collate_fn,
    )
    return train_loader, test_loader, train_cifar, test_cifar


# ============================================================================
# Training
# ============================================================================

def train_clip(model: CLIP, train_loader: DataLoader, test_loader: DataLoader):
    """학습 루프. epoch마다 train loss + retrieval 정확도 기록.

    시각화 03번을 위해 epoch 1 / mid / final 시점의 similarity matrix snapshot 저장.
    """
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98), eps=1e-6,
    )
    # Warmup + cosine decay
    warmup_steps = len(train_loader)  # 1 epoch warmup
    total_steps = len(train_loader) * EPOCHS

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = []
    snapshots = {}  # epoch → similarity matrix (32×32 sub-batch)
    snapshot_epochs = {1, EPOCHS // 2, EPOCHS}

    # 고정 snapshot batch (학습 진행 비교용)
    fixed_iter = iter(test_loader)
    fixed_batch = next(fixed_iter)
    fixed_images = fixed_batch[0][:32].to(DEVICE)
    fixed_tokens = fixed_batch[1][:32].to(DEVICE)
    fixed_masks = fixed_batch[2][:32].to(DEVICE)

    print(f"[TRAIN] Vocab size: {model.text_encoder.token_embed.num_embeddings}", flush=True)
    print(f"[TRAIN] Params - Image: {sum(p.numel() for p in model.image_encoder.parameters()):,}",
          flush=True)
    print(f"[TRAIN] Params - Text:  {sum(p.numel() for p in model.text_encoder.parameters()):,}",
          flush=True)
    print(f"[TRAIN] Params - Total: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_i2t = 0.0
        epoch_t2i = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            images, tokens, masks, _, _ = batch
            images = images.to(DEVICE)
            tokens = tokens.to(DEVICE)
            masks = masks.to(DEVICE)

            logits, _, _ = model(images, tokens, masks)
            loss = info_nce_loss(logits)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                i2t, t2i = retrieval_top1(logits)
            epoch_loss += loss.item()
            epoch_i2t += i2t
            epoch_t2i += t2i
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_i2t = epoch_i2t / n_batches
        avg_t2i = epoch_t2i / n_batches

        # snapshot 저장
        if epoch in snapshot_epochs:
            model.eval()
            with torch.no_grad():
                logits, _, _ = model(fixed_images, fixed_tokens, fixed_masks)
                snapshots[epoch] = logits.cpu().numpy()

        history.append({
            "epoch": epoch,
            "loss": avg_loss,
            "i2t_acc": avg_i2t,
            "t2i_acc": avg_t2i,
            "logit_scale": model.logit_scale.exp().item(),
        })
        elapsed = time.time() - t0
        print(
            f"[Epoch {epoch:2d}/{EPOCHS}] loss={avg_loss:.4f} | "
            f"i2t_acc={avg_i2t:.3f} t2i_acc={avg_t2i:.3f} | "
            f"τ={1.0/model.logit_scale.exp().item():.3f} | "
            f"{elapsed:.1f}s",
            flush=True,
        )

    return history, snapshots


# ============================================================================
# Checkpoint
# ============================================================================

def save_checkpoint(model: CLIP, tokenizer: Tokenizer, history: list, snapshots: dict):
    torch.save({
        "model": model.state_dict(),
        "vocab": tokenizer.idx_to_word,
        "history": history,
        "snapshots": snapshots,
    }, CHECKPOINT_PATH)
    print(f"[CKPT] Saved to {CHECKPOINT_PATH}", flush=True)


def load_checkpoint() -> tuple[CLIP, Tokenizer, list, dict]:
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    # 토크나이저 복원
    tokenizer = Tokenizer.__new__(Tokenizer)
    tokenizer.pad_idx = 0
    tokenizer.sos_idx = 1
    tokenizer.eos_idx = 2
    tokenizer.unk_idx = 3
    tokenizer.idx_to_word = ckpt["vocab"]
    tokenizer.word_to_idx = {w: i for i, w in enumerate(tokenizer.idx_to_word)}
    tokenizer.vocab_size = len(tokenizer.idx_to_word)
    # 모델 복원
    model = CLIP(vocab_size=tokenizer.vocab_size).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    return model, tokenizer, ckpt["history"], ckpt["snapshots"]


# ============================================================================
# Visualizations
# ============================================================================

def denormalize(image: torch.Tensor) -> np.ndarray:
    """정규화된 (C, H, W) tensor → (H, W, C) numpy [0, 1] for display."""
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    img = image.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def viz_01_dataset_overview(test_cifar, tokenizer: Tokenizer):
    """01. 데이터셋 개요 — CIFAR-10 이미지 + 합성 캡션 + 토큰 통계."""
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1], width_ratios=[2, 1, 1])

    # 좌측: 10개 클래스 샘플 이미지 + 캡션
    ax_img = fig.add_subplot(gs[0, :])
    ax_img.set_title("CIFAR-10 + 합성 캡션 샘플 (클래스당 1장)", fontsize=13, pad=10)
    ax_img.axis("off")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # 각 클래스에서 첫 샘플 가져오기
    sub_axes = []
    samples_per_class = {}
    for img, label in test_cifar:
        if label not in samples_per_class:
            samples_per_class[label] = img
        if len(samples_per_class) == 10:
            break

    inner = gs[0, :].subgridspec(2, 5, hspace=0.5)
    rng = np.random.RandomState(SEED)
    for cls_idx in range(10):
        ax = fig.add_subplot(inner[cls_idx // 5, cls_idx % 5])
        img = samples_per_class[cls_idx]
        ax.imshow(denormalize(img))
        cls_name = CIFAR10_CLASSES[cls_idx]
        tpl = TRAIN_TEMPLATES[rng.randint(0, len(TRAIN_TEMPLATES))]
        caption = tpl.format(cls=cls_name)
        ax.set_title(f'"{caption}"', fontsize=8)
        ax.axis("off")

    # 좌하: 클래스 분포
    ax_dist = fig.add_subplot(gs[1, 0])
    counts = np.zeros(10, dtype=int)
    for _, label in test_cifar:
        counts[label] += 1
    bars = ax_dist.bar(range(10), counts, color="steelblue", alpha=0.8)
    ax_dist.set_xticks(range(10))
    ax_dist.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right", fontsize=8)
    ax_dist.set_title("테스트셋 클래스별 샘플 수 (각 1,000장 균등)", fontsize=11)
    ax_dist.set_ylabel("Count")
    ax_dist.grid(axis="y", alpha=0.3)

    # 우하: 캡션 길이 분포 (토큰 수)
    ax_len = fig.add_subplot(gs[1, 1])
    lengths = []
    for tpl in TRAIN_TEMPLATES:
        for cls in CIFAR10_CLASSES:
            tokens, mask = tokenizer.encode(tpl.format(cls=cls))
            lengths.append(int(mask.sum().item()))
    ax_len.hist(lengths, bins=range(min(lengths), max(lengths) + 2),
                color="coral", alpha=0.8, edgecolor="black")
    ax_len.set_xlabel("토큰 수 (SOS/EOS 포함)")
    ax_len.set_ylabel("빈도")
    ax_len.set_title(f"캡션 토큰 길이 분포\n(평균 {np.mean(lengths):.1f})", fontsize=11)
    ax_len.grid(axis="y", alpha=0.3)

    # 우하: 어휘 통계
    ax_stat = fig.add_subplot(gs[1, 2])
    ax_stat.axis("off")
    stats_text = (
        f"[ 데이터셋 통계 ]\n\n"
        f"이미지: 50,000 (학습) / 10,000 (테스트)\n"
        f"해상도: 32 x 32 RGB\n"
        f"클래스: 10개 (각 균등 분포)\n\n"
        f"[ 캡션 통계 ]\n\n"
        f"학습 템플릿: {len(TRAIN_TEMPLATES)}개\n"
        f"평가 템플릿: {len(EVAL_TEMPLATES)}개 (zero-shot용)\n"
        f"어휘 크기: {tokenizer.vocab_size}\n"
        f"최대 토큰: {MAX_LEN}"
    )
    ax_stat.text(0.05, 0.95, stats_text, transform=ax_stat.transAxes,
                 fontsize=10, verticalalignment="top",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.7))

    plt.tight_layout()
    out = RESULTS_DIR / "01_dataset_overview.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] {out.name}", flush=True)


def viz_02_training_curve(history: list):
    """02. 학습 곡선 — InfoNCE Loss + Retrieval 정확도."""
    epochs = [h["epoch"] for h in history]
    losses = [h["loss"] for h in history]
    i2t = [h["i2t_acc"] for h in history]
    t2i = [h["t2i_acc"] for h in history]
    temps = [1.0 / h["logit_scale"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # (좌) InfoNCE Loss
    axes[0].plot(epochs, losses, "o-", color="darkblue", markersize=4, label="Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("InfoNCE Loss")
    axes[0].set_title("Symmetric InfoNCE Loss", fontsize=12)
    axes[0].grid(alpha=0.3)
    axes[0].axhline(y=math.log(BATCH_SIZE), color="gray", linestyle="--",
                    alpha=0.5, label=f"random baseline = ln({BATCH_SIZE}) ~ {math.log(BATCH_SIZE):.2f}")
    axes[0].legend(loc="upper right", fontsize=9)

    # (중) Retrieval Top-1 Accuracy
    axes[1].plot(epochs, i2t, "o-", color="crimson", markersize=4, label="Image → Text")
    axes[1].plot(epochs, t2i, "s-", color="darkgreen", markersize=4, label="Text → Image")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Top-1 Retrieval Acc")
    axes[1].set_title(f"Batch 내 Top-1 Retrieval 정확도 (Batch={BATCH_SIZE})", fontsize=12)
    axes[1].axhline(y=1.0 / BATCH_SIZE, color="gray", linestyle="--",
                    alpha=0.5, label=f"random 1/B = {1/BATCH_SIZE:.4f}")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="lower right", fontsize=9)

    # (우) Learnable Temperature
    axes[2].plot(epochs, temps, "o-", color="purple", markersize=4)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Temperature τ")
    axes[2].set_title("학습 가능한 Temperature (τ ↓ = 더 sharp한 분포)", fontsize=12)
    axes[2].axhline(y=INIT_TEMP, color="gray", linestyle="--",
                    alpha=0.5, label=f"초기값 τ={INIT_TEMP}")
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="upper right", fontsize=9)

    plt.suptitle("CLIP 학습 동학 — Loss · Retrieval Acc · Temperature", fontsize=13, y=1.02)
    plt.tight_layout()
    out = RESULTS_DIR / "02_training_curve.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] {out.name}", flush=True)


def viz_03_similarity_evolution(snapshots: dict):
    """03. Similarity Matrix 진화 — 학습 진행에 따라 대각선이 강해짐."""
    if not snapshots:
        print("[VIZ] No snapshots, skipping 03", flush=True)
        return

    epochs_sorted = sorted(snapshots.keys())
    n_panels = len(epochs_sorted)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    # 시각화용 정규화: 같은 컬러 스케일로 비교 가능하도록
    all_vals = np.concatenate([s.flatten() for s in snapshots.values()])
    vmin, vmax = np.percentile(all_vals, [5, 95])

    for ax, ep in zip(axes, epochs_sorted):
        sim = snapshots[ep]
        # 대각선 평균 vs 비대각선 평균 계산
        diag_mean = np.mean(np.diag(sim))
        off_diag_mean = (sim.sum() - np.diag(sim).sum()) / (sim.shape[0] * (sim.shape[0] - 1))
        gap = diag_mean - off_diag_mean

        im = ax.imshow(sim, cmap="RdYlBu_r", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(
            f"Epoch {ep} — Same-batch Similarity (32×32)\n"
            f"대각 평균 {diag_mean:.2f}  |  비대각 평균 {off_diag_mean:.2f}  |  격차 {gap:+.2f}",
            fontsize=10,
        )
        ax.set_xlabel("Text Index")
        ax.set_ylabel("Image Index")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(
        "Image × Text 유사도 행렬 진화 — 대각선(정답 짝)이 점점 강해지는 게 학습 신호",
        fontsize=12, y=1.05,
    )
    plt.tight_layout()
    out = RESULTS_DIR / "03_similarity_evolution.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] {out.name}", flush=True)


@torch.no_grad()
def compute_zero_shot(model: CLIP, tokenizer: Tokenizer, test_cifar, templates: list[str]):
    """학습 안 한 템플릿으로 zero-shot 분류 정확도 계산.

    템플릿 N개 × 클래스 10개 → 10개의 텍스트 임베딩 (템플릿 평균).
    각 이미지를 10개 텍스트와 유사도 비교 → argmax = 예측 클래스.
    """
    model.eval()
    # 클래스별 평균 텍스트 임베딩
    class_embs = []
    for cls_name in CIFAR10_CLASSES:
        cls_tokens = []
        cls_masks = []
        for tpl in templates:
            tk, mk = tokenizer.encode(tpl.format(cls=cls_name))
            cls_tokens.append(tk)
            cls_masks.append(mk)
        cls_tokens = torch.stack(cls_tokens).to(DEVICE)
        cls_masks = torch.stack(cls_masks).to(DEVICE)
        cls_emb = model.encode_text(cls_tokens, cls_masks).mean(dim=0)
        cls_emb = F.normalize(cls_emb, dim=-1)
        class_embs.append(cls_emb)
    class_embs = torch.stack(class_embs)   # (10, embed_dim)

    # 모든 테스트 이미지 평가
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    loader = DataLoader(test_cifar, batch_size=256, shuffle=False, num_workers=0)
    correct_per_class = np.zeros(10, dtype=int)
    total_per_class = np.zeros(10, dtype=int)
    correct_total = 0
    total = 0
    confusion = np.zeros((10, 10), dtype=int)

    for images, labels in loader:
        images = images.to(DEVICE)
        img_emb = model.encode_image(images)
        sims = img_emb @ class_embs.T   # (B, 10)
        preds = sims.argmax(dim=-1).cpu().numpy()
        labels_np = labels.numpy()
        for p, l in zip(preds, labels_np):
            confusion[l, p] += 1
            total_per_class[l] += 1
            if p == l:
                correct_per_class[l] += 1
                correct_total += 1
            total += 1

    accuracy = correct_total / total
    per_class_acc = correct_per_class / total_per_class.clip(min=1)
    return accuracy, per_class_acc, confusion


def viz_04_zero_shot(model: CLIP, tokenizer: Tokenizer, test_cifar):
    """04. Zero-shot 분류 — 학습 때 못 본 템플릿으로 분류 성능 측정."""
    acc_train, per_cls_train, _ = compute_zero_shot(model, tokenizer, test_cifar, TRAIN_TEMPLATES)
    acc_eval, per_cls_eval, conf_eval = compute_zero_shot(model, tokenizer, test_cifar, EVAL_TEMPLATES)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # (좌) 클래스별 정확도 bar chart
    x = np.arange(10)
    width = 0.38
    axes[0].bar(x - width / 2, per_cls_train * 100, width, label=f"학습 템플릿 (전체 {acc_train*100:.1f}%)",
                color="steelblue", alpha=0.8)
    axes[0].bar(x + width / 2, per_cls_eval * 100, width, label=f"Zero-shot 템플릿 (전체 {acc_eval*100:.1f}%)",
                color="coral", alpha=0.8)
    axes[0].axhline(y=10, color="gray", linestyle="--", alpha=0.5, label="random 10%")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right", fontsize=9)
    axes[0].set_ylabel("정확도 (%)")
    axes[0].set_title("클래스별 Zero-shot 분류 정확도\n학습 템플릿 vs 학습 때 안 본 새 템플릿", fontsize=11)
    axes[0].legend(fontsize=9, loc="lower right")
    axes[0].grid(axis="y", alpha=0.3)

    # (우) Confusion matrix (zero-shot 템플릿 기준)
    im = axes[1].imshow(conf_eval, cmap="Blues", aspect="auto")
    axes[1].set_xticks(range(10))
    axes[1].set_yticks(range(10))
    axes[1].set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right", fontsize=8)
    axes[1].set_yticklabels(CIFAR10_CLASSES, fontsize=8)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title(f"Zero-shot Confusion Matrix\n(평가 템플릿 기준, 전체 {acc_eval*100:.1f}%)",
                      fontsize=11)
    # 셀별 숫자 표기
    for i in range(10):
        for j in range(10):
            color = "white" if conf_eval[i, j] > conf_eval.max() / 2 else "black"
            axes[1].text(j, i, conf_eval[i, j], ha="center", va="center",
                         fontsize=7, color=color)
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.suptitle(
        "Zero-shot 분류 — 학습 때 못 본 캡션 템플릿으로 분류 (CLIP의 핵심 능력)",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    out = RESULTS_DIR / "04_zero_shot.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] {out.name} | train_tpl={acc_train*100:.2f}% | zero_shot={acc_eval*100:.2f}%",
          flush=True)
    return acc_train, acc_eval


@torch.no_grad()
def viz_05_embedding_space(model: CLIP, tokenizer: Tokenizer, test_cifar):
    """05. Multimodal Embedding Space (t-SNE) — 같은 클래스 이미지+텍스트가 가까이 있는지."""
    model.eval()
    # 클래스당 20장 이미지 샘플
    samples_per_class = {i: [] for i in range(10)}
    for img, label in test_cifar:
        if len(samples_per_class[label]) < 20:
            samples_per_class[label].append(img)
        if all(len(v) == 20 for v in samples_per_class.values()):
            break

    # 이미지 임베딩
    all_imgs = torch.stack([img for lst in samples_per_class.values() for img in lst])
    img_labels = np.array([cls for cls in range(10) for _ in range(20)])
    all_imgs = all_imgs.to(DEVICE)
    img_embs = model.encode_image(all_imgs).cpu().numpy()

    # 클래스별 텍스트 임베딩 (학습 + 평가 템플릿 모두)
    text_embs = []
    text_labels = []
    for cls_idx, cls_name in enumerate(CIFAR10_CLASSES):
        for tpl in TRAIN_TEMPLATES + EVAL_TEMPLATES:
            tk, mk = tokenizer.encode(tpl.format(cls=cls_name))
            emb = model.encode_text(tk.unsqueeze(0).to(DEVICE), mk.unsqueeze(0).to(DEVICE))
            text_embs.append(emb.squeeze(0).cpu().numpy())
            text_labels.append(cls_idx)
    text_embs = np.array(text_embs)
    text_labels = np.array(text_labels)

    # 합쳐서 t-SNE
    all_embs = np.concatenate([img_embs, text_embs], axis=0)
    n_img = len(img_embs)
    tsne = TSNE(n_components=2, perplexity=30, random_state=SEED, init="pca")
    coords = tsne.fit_transform(all_embs)

    img_coords = coords[:n_img]
    text_coords = coords[n_img:]

    fig, ax = plt.subplots(figsize=(11, 9))
    cmap = plt.cm.tab10
    # 이미지: 채워진 원
    for cls in range(10):
        m = img_labels == cls
        ax.scatter(img_coords[m, 0], img_coords[m, 1], c=[cmap(cls)],
                   s=50, alpha=0.6, marker="o", label=f"{CIFAR10_CLASSES[cls]} (image)")
    # 텍스트: 큰 X
    for cls in range(10):
        m = text_labels == cls
        ax.scatter(text_coords[m, 0], text_coords[m, 1], c=[cmap(cls)],
                   s=200, alpha=0.95, marker="X", edgecolors="black", linewidths=1.5)

    ax.set_title(
        "Multimodal Embedding Space (t-SNE) — 이미지(원)와 텍스트(X)가 같은 클래스끼리 모이는지\n"
        "같은 색 원과 X가 가까이 있으면 multimodal alignment 성공",
        fontsize=12,
    )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    # 범례는 클래스만 (modal은 별도 표기)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap(c),
                          markersize=10, label=CIFAR10_CLASSES[c]) for c in range(10)]
    handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                              markersize=10, label="Image (원)"))
    handles.append(plt.Line2D([0], [0], marker="X", color="w", markerfacecolor="gray",
                              markersize=14, markeredgecolor="black", label="Text (X)"))
    ax.legend(handles=handles, loc="best", fontsize=9, ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = RESULTS_DIR / "05_embedding_space.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] {out.name}", flush=True)


@torch.no_grad()
def viz_06_retrieval_examples(model: CLIP, tokenizer: Tokenizer, test_cifar):
    """06. Text → Image Retrieval — 캡션 쿼리로 가장 유사한 이미지 top-5 검색."""
    model.eval()
    # 테스트셋 전체 이미지 임베딩 한 번에 계산 (5000장 정도로 제한)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    images = []
    labels = []
    for i, (img, lab) in enumerate(test_cifar):
        images.append(img)
        labels.append(lab)
        if i >= 4999:
            break
    images_t = torch.stack(images).to(DEVICE)
    img_embs = model.encode_image(images_t).cpu().numpy()
    labels = np.array(labels)

    # 쿼리: 클래스마다 1개씩, 평가 템플릿으로 (zero-shot 검증 의미)
    queries = []
    for cls_name in CIFAR10_CLASSES:
        queries.append(f"a photo of a {cls_name}")

    fig, axes = plt.subplots(len(queries), 6, figsize=(13, 1.7 * len(queries)))
    for q_idx, query in enumerate(queries):
        tk, mk = tokenizer.encode(query)
        txt_emb = model.encode_text(tk.unsqueeze(0).to(DEVICE),
                                     mk.unsqueeze(0).to(DEVICE)).cpu().numpy()
        sims = (img_embs @ txt_emb.T).flatten()
        top5_idx = sims.argsort()[-5:][::-1]

        # 좌측: 쿼리 텍스트
        axes[q_idx, 0].axis("off")
        axes[q_idx, 0].text(0.05, 0.5, f'"{query}"',
                            transform=axes[q_idx, 0].transAxes,
                            fontsize=10, verticalalignment="center",
                            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow"))

        # 우측: top-5 이미지
        for rank, idx in enumerate(top5_idx):
            ax = axes[q_idx, rank + 1]
            ax.imshow(denormalize(images[idx]))
            pred_cls = CIFAR10_CLASSES[labels[idx]]
            true_match = (labels[idx] == q_idx)
            color = "green" if true_match else "red"
            mark = "[O]" if true_match else "[X]"
            ax.set_title(f"{mark} {pred_cls}\n(sim={sims[idx]:.2f})",
                         fontsize=8, color=color)
            ax.axis("off")
            # 테두리 색
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(color)
                spine.set_linewidth(2)

    plt.suptitle(
        "Text → Image Retrieval — 캡션 쿼리로 가장 유사한 이미지 top-5\n"
        "초록 [O] = 같은 클래스 (정답), 빨강 [X] = 다른 클래스",
        fontsize=13, y=1.005,
    )
    plt.tight_layout()
    out = RESULTS_DIR / "06_retrieval_examples.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] {out.name}", flush=True)


# ============================================================================
# CLI + Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="CLIP from Scratch on CIFAR-10")
    parser.add_argument("--retrain", action="store_true",
                        help="체크포인트 무시하고 처음부터 재학습")
    parser.add_argument("--viz-only", action="store_true",
                        help="학습 건너뛰고 체크포인트로 시각화만 생성")
    return parser.parse_args()


def main():
    args = parse_args()

    # 토크나이저는 항상 동일하게 빌드 (학습/추론 모두)
    tokenizer = build_tokenizer()
    print(f"[INIT] Tokenizer vocab size: {tokenizer.vocab_size}", flush=True)

    # 학습 (필요 시)
    if args.viz_only:
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"--viz-only인데 체크포인트가 없음: {CHECKPOINT_PATH}")
        model, tokenizer, history, snapshots = load_checkpoint()
        print(f"[LOAD] {len(history)} epoch history loaded", flush=True)
    elif not args.retrain and CHECKPOINT_PATH.exists():
        print(f"[LOAD] 기존 체크포인트 사용: {CHECKPOINT_PATH} (재학습 원하면 --retrain)", flush=True)
        model, tokenizer, history, snapshots = load_checkpoint()
    else:
        train_loader, test_loader, _, _ = get_dataloaders(tokenizer)
        model = CLIP(vocab_size=tokenizer.vocab_size).to(DEVICE)
        history, snapshots = train_clip(model, train_loader, test_loader)
        save_checkpoint(model, tokenizer, history, snapshots)

    # 시각화용 테스트셋
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    test_cifar = torchvision.datasets.CIFAR10(
        str(DATA_DIR), train=False, download=True, transform=transform,
    )

    # 시각화 6개
    print("\n=== Visualization ===", flush=True)
    viz_01_dataset_overview(test_cifar, tokenizer)
    viz_02_training_curve(history)
    viz_03_similarity_evolution(snapshots)
    viz_04_zero_shot(model, tokenizer, test_cifar)
    viz_05_embedding_space(model, tokenizer, test_cifar)
    viz_06_retrieval_examples(model, tokenizer, test_cifar)

    print(f"\n=== Done. Results in {RESULTS_DIR} ===", flush=True)


if __name__ == "__main__":
    main()
