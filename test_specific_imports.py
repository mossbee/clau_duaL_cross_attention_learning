"""
Test specific imports that were failing on Kaggle
"""

def test_specific_failing_imports():
    """Test the exact imports that were causing NameError on Kaggle"""
    print("Testing specific failing imports from Kaggle...")
    
    # Test 1: models import (was failing with List not defined)
    print("✓ Testing models import...")
    try:
        from dual_cross_attention.models import DualCrossAttentionViT
        print("  ✅ DualCrossAttentionViT imported successfully")
    except Exception as e:
        print(f"  ❌ DualCrossAttentionViT import failed: {e}")
        return False
    
    # Test 2: datasets import (was failing with FGVCDataLoader not found)  
    print("✓ Testing datasets import...")
    try:
        from dual_cross_attention.datasets import FGVCDataLoader, ReIDDataLoader
        print("  ✅ FGVCDataLoader and ReIDDataLoader imported successfully")
    except Exception as e:
        print(f"  ❌ DataLoader imports failed: {e}")
        return False
    
    # Test 3: utils import (latest failure - List not defined in loss_functions)
    print("✓ Testing utils import...")
    try:
        from dual_cross_attention.utils import (
            DualCrossAttentionLoss, UncertaintyWeightedLoss, TripletLoss, 
            FGVCMetrics, AttentionVisualizer
        )
        print("  ✅ Utils imports successful")
    except Exception as e:
        print(f"  ❌ Utils imports failed: {e}")
        return False
    
    # Test 4: Individual file imports that were problematic
    print("✓ Testing individual file imports...")
    try:
        from dual_cross_attention.utils.loss_functions import UncertaintyWeightedLoss, TripletLoss, CrossEntropyLoss
        print("  ✅ loss_functions imports successful")
    except Exception as e:
        print(f"  ❌ loss_functions imports failed: {e}")
        return False
    
    try:
        from dual_cross_attention.models.vit_backbone import VisionTransformer
        print("  ✅ vit_backbone imports successful")
    except Exception as e:
        print(f"  ❌ vit_backbone imports failed: {e}")
        return False
    
    try:
        from dual_cross_attention.models.attention_modules import SelfAttention, GlobalLocalCrossAttention, PairWiseCrossAttention
        print("  ✅ attention_modules imports successful")
    except Exception as e:
        print(f"  ❌ attention_modules imports failed: {e}")
        return False
    
    print("✅ All specific failing imports now work!")
    return True


def test_train_script_imports():
    """Test the exact imports from train.py that were failing"""
    print("\nTesting train.py imports...")
    
    try:
        # Line 37 in train.py
        from dual_cross_attention.models import DualCrossAttentionViT
        
        # Line 38-42 in train.py  
        from dual_cross_attention.datasets import FGVCDataLoader, ReIDDataLoader
        from dual_cross_attention.utils import (
            DualCrossAttentionLoss, FGVCMetrics, AttentionVisualizer, MetricsTracker
        )
        from dual_cross_attention.configs import FGVCConfig, ReIDConfig
        
        # Latest failing import - line 43
        from dual_cross_attention.configs import get_fgvc_config, get_reid_config
        
        print("✅ All train.py imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ train.py imports failed: {e}")
        return False


if __name__ == "__main__":
    print("="*70)
    print("TESTING SPECIFIC KAGGLE IMPORT FAILURES")
    print("="*70)
    
    success = True
    success &= test_specific_failing_imports()
    success &= test_train_script_imports()
    
    print("\n" + "="*70)
    if success:
        print("🎉 ALL KAGGLE IMPORT ISSUES RESOLVED!")
        print("Your code should now run without import errors on Kaggle.")
    else:
        print("❌ Some import issues remain.")
    print("="*70)
