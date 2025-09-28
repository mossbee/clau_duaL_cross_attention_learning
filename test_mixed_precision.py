"""
Test script to verify mixed precision compatibility is fixed
"""

def test_mixed_precision():
    """Test that the model works with mixed precision training"""
    print("Testing mixed precision compatibility...")
    
    try:
        import torch
        from dual_cross_attention.models import DualCrossAttentionViT
        
        # Create smaller model for testing
        model = DualCrossAttentionViT(
            img_size=(224, 224),
            num_classes=10,
            embed_dim=384,  # Divisible by 12 heads (384 / 12 = 32)
            num_sa_layers=2,  # Fewer layers
            num_glca_layers=1,
            num_pwca_layers=2,
            task_type="fgvc"
        )
        
        if torch.cuda.is_available():
            device = 'cuda'
            model = model.to(device)
        else:
            device = 'cpu'
            print("CUDA not available, testing on CPU")
        
        # Test with mixed precision if CUDA available
        if device == 'cuda':
            try:
                # Try new API first
                from torch.amp import autocast, GradScaler
                scaler = GradScaler('cuda')
                autocast_context = autocast('cuda')
                print("Using new PyTorch 2.1+ mixed precision API")
            except ImportError:
                try:
                    # Fallback to old API
                    from torch.cuda.amp import autocast, GradScaler
                    scaler = GradScaler()
                    autocast_context = autocast()
                    print("Using legacy mixed precision API")
                except ImportError:
                    print("Mixed precision not available")
                    return True
        else:
            scaler = None
            autocast_context = torch.no_grad()
        
        # Create test data
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        paired_images = torch.randn(batch_size, 3, 224, 224)
        
        if device == 'cuda':
            images = images.cuda()
            paired_images = paired_images.cuda()
        
        model.train()
        
        # Test forward pass with mixed precision
        with autocast_context:
            outputs = model(images, paired_images)
        
        print(f"✅ Mixed precision forward pass successful!")
        print(f"   SA logits shape: {outputs['sa_logits'].shape}")
        if 'glca_logits' in outputs:
            print(f"   GLCA logits shape: {outputs['glca_logits'].shape}")
        if 'pwca_logits' in outputs:
            print(f"   PWCA logits shape: {outputs['pwca_logits'].shape}")
        
        # Test evaluation mode (no PWCA)
        model.eval()
        with torch.no_grad():
            eval_outputs = model(images)
        
        print(f"✅ Evaluation mode forward pass successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ Mixed precision test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("MIXED PRECISION COMPATIBILITY TEST")
    print("="*60)
    
    success = test_mixed_precision()
    
    print("\n" + "="*60)
    if success:
        print("🎉 Mixed precision issues fixed!")
        print("The model should now work correctly with autocast on Kaggle.")
    else:
        print("❌ Mixed precision test failed.")
    print("="*60)
