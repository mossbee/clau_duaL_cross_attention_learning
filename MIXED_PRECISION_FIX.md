# Mixed Precision Dtype Fix

## Issue

When training with mixed precision (FP16) enabled, a dtype mismatch error occurred in the GLCA module:

```
RuntimeError: Index put requires the source and destination dtypes match, 
got Float for the destination and Half for the source.
```

## Root Cause

In `attention_modules.py`, the GLCA forward method created an output tensor using:
```python
output = torch.zeros_like(x)  # Uses x's dtype (may be FP32)
```

But then tried to assign values from attention operations:
```python
output[b, selected_indices[b]] = out[b]  # out is in FP16 due to autocast
```

When mixed precision is enabled:
- `x` may be in FP32 (the input before autocast context)
- `out` is in FP16 (computed within autocast context)
- PyTorch doesn't allow assigning FP16 values to FP32 tensor

## Solution

Changed the output tensor creation to match the dtype of the attention output:

```python
# Before
output = torch.zeros_like(x)

# After
output = torch.zeros_like(x, dtype=out.dtype)
```

This ensures both tensors have the same dtype, avoiding the mismatch error.

## File Modified

- `dual_cross_attention/models/attention_modules.py` (line 237)

## Impact

- ✅ Fixes mixed precision training
- ✅ No performance impact
- ✅ No algorithmic changes
- ✅ Maintains paper compliance

## Verification

The fix has been tested and works correctly with:
- Mixed precision (FP16) training ✅
- Gradient checkpointing ✅
- Full model forward/backward pass ✅

## Status

✅ **FIXED** - Training should now proceed without dtype errors.

