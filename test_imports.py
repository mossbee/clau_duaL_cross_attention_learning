"""
Test script to verify all imports work correctly.
Run this to check if the implementation can be imported without errors.
"""

def test_imports():
    """Test all major imports"""
    print("Testing imports...")
    
    try:
        print("✓ Testing basic imports...")
        import torch
        import numpy as np
        
        print("✓ Testing dual_cross_attention.models...")
        from dual_cross_attention.models import DualCrossAttentionViT
        
        print("✓ Testing dual_cross_attention.utils...")
        from dual_cross_attention.utils import DualCrossAttentionLoss, FGVCMetrics, AttentionVisualizer
        
        print("✓ Testing dual_cross_attention.datasets...")
        from dual_cross_attention.datasets import CUBDataset
        from dual_cross_attention.datasets.transforms import FGVCTransforms
        
        print("✓ All core imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_model_creation():
    """Test model creation"""
    print("\nTesting model creation...")
    
    try:
        from dual_cross_attention.models import DualCrossAttentionViT
        
        model = DualCrossAttentionViT(
            img_size=(224, 224),
            num_classes=10,
            embed_dim=384,
            num_sa_layers=6,
            num_glca_layers=1,
            num_pwca_layers=6,
            task_type="fgvc"
        )
        
        print(f"✓ Model created with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
        return True
        
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False

def test_forward_pass():
    """Test forward pass"""
    print("\nTesting forward pass...")
    
    try:
        from dual_cross_attention.models import DualCrossAttentionViT
        import torch
        
        model = DualCrossAttentionViT(
            img_size=(224, 224),
            num_classes=10,
            embed_dim=384,
            num_sa_layers=6,
            num_glca_layers=1,
            num_pwca_layers=6,
            task_type="fgvc"
        )
        
        # Test evaluation mode (no PWCA)
        model.eval()
        with torch.no_grad():
            x = torch.randn(2, 3, 224, 224)
            outputs = model(x)
        
        print(f"✓ Forward pass successful. Outputs: {list(outputs.keys())}")
        return True
        
    except Exception as e:
        print(f"❌ Forward pass error: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("DUAL CROSS-ATTENTION LEARNING - IMPORT TEST")
    print("="*60)
    
    success = True
    success &= test_imports()
    success &= test_model_creation()
    success &= test_forward_pass()
    
    print("\n" + "="*60)
    if success:
        print("🎉 ALL TESTS PASSED! Implementation is ready to use.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    print("="*60)
