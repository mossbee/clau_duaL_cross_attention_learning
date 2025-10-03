# Variable Name Conflict Fix

## Issue

Training failed with the error:
```
AttributeError: 'DualCrossAttentionBlock' object has no attribute 'pwca'
```

## Root Cause

There was a **variable name conflict** in the model's forward method:

1. **SA/PWCA section** (lines 406-463): Uses `for block in self.sa_pwca_blocks:`
   - These are `TransformerBlockWithPWCA` objects that **have** `pwca` attribute

2. **GLCA section** (lines 486-489): Uses `for block in self.glca_blocks:`
   - These are `DualCrossAttentionBlock` objects that **don't have** `pwca` attribute

The problem: The `block` variable was **overwritten** by the GLCA loop, so when the gradient checkpointing code tried to access `block.pwca`, it was actually accessing a `DualCrossAttentionBlock` instead of a `TransformerBlockWithPWCA`.

## Solution

Changed the variable name in the GLCA section to avoid conflict:

```python
# Before (caused conflict)
for block in self.glca_blocks:
    block_outputs = block(glca_x, attention_history=sa_attention_history)

# After (fixed)
for glca_block in self.glca_blocks:
    block_outputs = glca_block(glca_x, attention_history=sa_attention_history)
```

## File Modified

- `dual_cross_attention/models/dual_vit.py` (line 486)

## Impact

- ✅ Fixes gradient checkpointing with PWCA
- ✅ No performance impact
- ✅ No algorithmic changes
- ✅ Maintains paper compliance

## Verification

The fix has been tested and works correctly:
- ✅ Forward pass with PWCA (training mode)
- ✅ All output keys present (SA, PWCA, GLCA)
- ✅ Gradient checkpointing functional
- ✅ No variable name conflicts

## Status

✅ **FIXED** - Training should now proceed without variable name conflicts.

## Summary of All Fixes

1. **OOM Fix**: Reduced batch size, added gradient checkpointing
2. **Mixed Precision Fix**: Fixed dtype mismatch in GLCA module  
3. **Variable Conflict Fix**: Fixed variable name collision between SA/PWCA and GLCA sections

All fixes maintain strict adherence to the paper specifications while enabling training on 16GB GPUs.
