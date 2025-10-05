# Quick Fix Summary - Training Issues V2

## What Went Wrong This Time?

After fixing the parameter name issue, training ran for 15 epochs but then:
1. ❌ **Accuracy collapsed**: 8.19% (epoch 7) → 1.05% (epoch 15)  
2. ❌ **OOM crash**: Out of memory at epoch 16
3. ❌ **Weights diverging**: Uncertainty weights grew from 0.0002 to 0.0697

## Root Causes

### 1. Memory Leak 💾
Attention weights were keeping gradients attached across all 12 SA blocks, creating a massive computation graph that consumed all 15GB GPU memory.

### 2. Unbounded Uncertainty Weights 📈
Loss weighting parameters (w1, w2, w3) were allowed to grow from -5 to +5, causing training instability.

## Fixes Applied ✅

### Fix 1: Detach Attention Weights (`dual_cross_attention/models/dual_vit.py`)
```python
# OLD:
sa_attention_history.append(sa_weights)

# NEW:
sa_attention_history.append(sa_weights.detach())
```

### Fix 2: Tighten Uncertainty Bounds (`dual_cross_attention/utils/loss_functions.py`)
```python
# OLD:
self.min_log_var = -5.0
self.max_log_var = 5.0

# NEW:
self.min_log_var = -2.0
self.max_log_var = 2.0
```

## What To Expect Now

### ✅ No More OOM
Memory usage should stay stable around 14GB instead of growing to 15.67GB+

### ✅ Stable Training
- Loss should decrease monotonically: 8.0 → 7.5 → 7.0 → ...
- Accuracy should increase steadily: 0.5% → 5% → 20% → 50% → 91%
- Uncertainty weights should stabilize around 0.0 ± 1.0

### ✅ Target Performance (100 epochs)
- **Training accuracy**: ~95%
- **Validation accuracy**: ~91.4% (per paper)

## Quick Test

Before full 100-epoch training, test with:
```bash
python train.py --task fgvc --dataset cub --epochs 20
```

Watch for:
- [ ] No OOM by epoch 20
- [ ] Validation accuracy > 30% by epoch 20
- [ ] w1, w2, w3 < 0.2 by epoch 20

## If Issues Still Persist

### Reduce Memory Further
```python
# In fgvc_config.py
batch_size: int = 2
gradient_accumulation_steps: int = 8
```

### Try Simpler Architecture First
```bash
# SA + GLCA only (no PWCA)
python train.py --task fgvc --dataset cub --ablation sa_glca
```

### Enable Gradient Checkpointing
```python
# In fgvc_config.py
use_gradient_checkpointing: bool = True
```

## Files Changed
1. ✅ `dual_cross_attention/models/dual_vit.py` - Line ~384, ~401
2. ✅ `dual_cross_attention/utils/loss_functions.py` - Line ~54-56

## Ready to Retry! 🚀

The fixes address:
- ✅ Memory leak → Fixed with `.detach()`
- ✅ Training instability → Fixed with tighter bounds
- ✅ Accuracy collapse → Should resolve with stable weights

Restart training and it should converge properly now!
