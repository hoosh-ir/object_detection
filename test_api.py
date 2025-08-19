#!/usr/bin/env python3
"""
Simple test script to validate the API components before Docker build
"""

import sys
import traceback

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import mmcv
        print("✓ mmcv imported successfully")
    except ImportError as e:
        print(f"✗ mmcv import failed: {e}")
        return False
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} imported successfully")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device count: {torch.cuda.device_count()}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        from mmdet3d.online_inference_plugin.inference_api import InferenceLidarAPI, InferenceCameraAPI
        print("✓ Inference APIs imported successfully")
    except ImportError as e:
        print(f"✗ Inference API import failed: {e}")
        return False
    
    try:
        import fastapi
        print(f"✓ FastAPI {fastapi.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ FastAPI import failed: {e}")
        return False
        
    return True

def test_model_configs():
    """Test if model configurations are accessible"""
    print("\nTesting model configurations...")
    
    try:
        from mmdet3d.online_inference_plugin.inference_api import model_to_config, model_to_checkpoint
        
        for model_name in ["pointpillars", "second", "imvoxelnet"]:
            if model_name in model_to_config:
                config_path = model_to_config[model_name]
                checkpoint_path = model_to_checkpoint[model_name]
                print(f"✓ {model_name}: config={config_path}, checkpoint={checkpoint_path}")
            else:
                print(f"✗ {model_name}: not found in model configs")
                return False
                
        return True
    except Exception as e:
        print(f"✗ Model config test failed: {e}")
        traceback.print_exc()
        return False

def test_checkpoint_files():
    """Test if checkpoint files exist"""
    print("\nTesting checkpoint files...")
    
    import os
    from mmdet3d.online_inference_plugin.inference_api import model_to_checkpoint
    
    all_exist = True
    for model_name, checkpoint_path in model_to_checkpoint.items():
        if os.path.isfile(checkpoint_path):
            file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
            print(f"✓ {model_name}: {checkpoint_path} ({file_size:.1f} MB)")
        else:
            print(f"✗ {model_name}: {checkpoint_path} not found")
            all_exist = False
            
    return all_exist

def test_data_loading():
    """Test data loading functions"""
    print("\nTesting data loading functions...")
    
    try:
        from mmdet3d.online_inference_plugin.data import load_pcd, load_image
        print("✓ Data loading functions imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Data loading import failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=== API Component Validation ===\n")
    
    tests = [
        test_imports,
        test_model_configs,
        test_checkpoint_files,
        test_data_loading
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"Test {test.__name__} failed")
        except Exception as e:
            print(f"Test {test.__name__} crashed: {e}")
            traceback.print_exc()
    
    print(f"\n=== Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("✓ All tests passed! Ready for Docker build.")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
