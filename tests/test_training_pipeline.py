"""
Smoke test — verifies all training modules import and run correctly.
Run: python tests/test_training_pipeline.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision
import numpy as np

def header(text):
    print(f"\n{'='*55}")
    print(f"  {text}")
    print(f"{'='*55}")

def ok(msg):
    print(f"  [PASS] {msg}")

def section(msg):
    print(f"\n  --- {msg} ---")

# ── 1. Env ────────────────────────────────────────────────────────────────
header("Environment")
print(f"  PyTorch:      {torch.__version__}")
print(f"  torchvision:  {torchvision.__version__}")
print(f"  CUDA:         {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device:       {device}")

# ── 2. Imports ────────────────────────────────────────────────────────────
header("Module Imports")
from src.models.efficientnet import build_model, EfficientNetB0
ok("src.models.efficientnet")

from src.training.loss import FocalLoss, build_loss
ok("src.training.loss")

from src.training.metrics import MetricAccumulator, compute_metrics, find_optimal_threshold
ok("src.training.metrics")

from src.training.early_stopping import EarlyStopping
ok("src.training.early_stopping")

from src.training.checkpoint import CheckpointManager
ok("src.training.checkpoint")

from src.data.dataset import ALLDataset
ok("src.data.dataset")

from src.data.augmentation import get_train_transforms, get_val_transforms
ok("src.data.augmentation")

# ── 3. Model ──────────────────────────────────────────────────────────────
header("Model Construction & Phase Management")
model = build_model(pretrained=False, dropout=0.4, num_classes=1, device=device)
ok("build_model(pretrained=False)")

total = sum(p.numel() for p in model.parameters())

section("Phase 1 — head only")
model.set_phase(1)
tr1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
ok(f"trainable: {tr1:,} / {total:,}  ({100*tr1/total:.1f}%)")
assert tr1 < total * 0.20, "Phase 1 should have <20% trainable (head only)"

section("Phase 2 — top blocks")
model.set_phase(2)
tr2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
ok(f"trainable: {tr2:,} / {total:,}  ({100*tr2/total:.1f}%)")
assert tr1 < tr2 < total, "Phase 2 must have more trainable than Phase 1"

section("Phase 3 — full fine-tune")
model.set_phase(3)
tr3 = sum(p.numel() for p in model.parameters() if p.requires_grad)
ok(f"trainable: {tr3:,} / {total:,}  ({100*tr3/total:.1f}%)")
assert tr3 >= total * 0.98, f"Phase 3 must unfreeze nearly all params (BN stays frozen), got {100*tr3/total:.1f}%"

# ── 4. Forward Pass ───────────────────────────────────────────────────────
header("Forward Pass")
model.eval()
model.set_phase(1)
x = torch.randn(4, 3, 224, 224).to(device)
with torch.no_grad():
    out = model(x)
assert out.shape == (4, 1), f"Expected (4,1), got {out.shape}"
ok(f"Input: {list(x.shape)}  ->  Output: {list(out.shape)}")

# ── 5. Loss Functions ─────────────────────────────────────────────────────
header("Loss Functions")
labels = torch.tensor([1.0, 0.0, 1.0, 0.0]).to(device)

focal = build_loss("focal", focal_alpha=0.25, focal_gamma=2.0, label_smoothing=0.1)
fl = focal(out.squeeze(-1), labels)
assert not torch.isnan(fl), "FocalLoss returned NaN"
ok(f"FocalLoss: {fl.item():.6f}")

bce = build_loss("bce")
bl = bce(out.squeeze(-1), labels)
ok(f"BCELoss:   {bl.item():.6f}")

bce_w = build_loss("bce_weighted", pos_weight=1.64)
bw = bce_w(out.squeeze(-1), labels)
ok(f"BCEWeighted: {bw.item():.6f}")

# ── 6. Metrics ────────────────────────────────────────────────────────────
header("MetricAccumulator")
acc = MetricAccumulator()
acc.update(out.detach().cpu(), labels.cpu(), float(fl.item()))
# Add a second batch
out2 = torch.randn(4, 1)
lbl2 = torch.tensor([0.0, 1.0, 1.0, 0.0])
acc.update(out2, lbl2, 0.45)

metrics = acc.compute()
ok(f"loss={metrics['loss']:.4f}  auc={metrics['auc_roc']:.4f}  f1={metrics['f1']:.4f}")
ok(f"sensitivity={metrics['sensitivity']:.4f}  specificity={metrics['specificity']:.4f}")
ok(f"optimal_threshold={metrics['optimal_threshold']:.4f}  youden_j={metrics['youden_j']:.4f}")

# ── 7. Early Stopping ─────────────────────────────────────────────────────
header("Early Stopping")
es = EarlyStopping(monitor="val_auc", patience=3, mode="max", verbose=False)
model.set_phase(1)
es.step(0.80, 1, model)
es.step(0.85, 2, model)
es.step(0.83, 3, model)
es.step(0.82, 4, model)
stopped = es.step(0.81, 5, model)

ok(f"best_value={es.best_value:.3f}  best_epoch={es.best_epoch}  counter={es.counter}  stopped={stopped}")
assert es.best_epoch == 2, f"Best epoch should be 2, got {es.best_epoch}"
assert stopped, "Should have triggered patience=3 after 3 non-improving epochs"

# ── 8. Augmentation Pipelines ─────────────────────────────────────────────
header("Augmentation Pipelines")
train_tfm = get_train_transforms(224)
val_tfm   = get_val_transforms(224)
img_np    = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

aug_result  = train_tfm(image=img_np)
val_result  = val_tfm(image=img_np)
assert aug_result["image"].shape == (224, 224, 3), "Train transform output shape mismatch"
assert val_result["image"].shape == (224, 224, 3), "Val transform output shape mismatch"
ok(f"Train transform: {img_np.shape} -> {aug_result['image'].shape}")
ok(f"Val   transform: {img_np.shape} -> {val_result['image'].shape}")

# ── 9. CheckpointManager ─────────────────────────────────────────────────
header("CheckpointManager (save/load)")
import tempfile
ckpt = CheckpointManager(
    checkpoint_dir="tests/tmp_ckpt",
    monitor="auc_roc",
    mode="max",
    filename="test_best.pth",
)
optimizer  = torch.optim.AdamW(model.parameters(), lr=1e-3)
from src.training.trainer import CosineWarmupScheduler
scheduler  = CosineWarmupScheduler(optimizer)
es2        = EarlyStopping(patience=5, verbose=False)

is_best = ckpt.save(
    epoch=1, model=model, optimizer=optimizer,
    scheduler=scheduler, early_stopping=es2,
    metrics={"auc_roc": 0.92}, history={},
    current_phase=1,
)
assert is_best, "First save should always be best"
ok(f"Checkpoint saved: {ckpt.best_path}")

state = ckpt.load()
assert state["epoch"] == 1
ok(f"Checkpoint loaded: epoch={state['epoch']}, auc={state['current_metrics'].get('auc_roc', 0):.2f}")

# Cleanup
import shutil
shutil.rmtree("tests/tmp_ckpt", ignore_errors=True)

# ── Summary ───────────────────────────────────────────────────────────────
header("ALL SMOKE TESTS PASSED")
print(f"  Device: {device}")
print(f"  PyTorch: {torch.__version__}")
print(f"  Model params: {total:,}")
print()
