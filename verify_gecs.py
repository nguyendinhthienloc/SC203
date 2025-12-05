#!/usr/bin/env python
"""
GECS Integration Verification Script

This script verifies that all GECS components are properly integrated
and functional.
"""

import sys
from pathlib import Path

def verify_imports():
    """Verify all required imports work."""
    print("=" * 60)
    print("GECS Integration Verification")
    print("=" * 60)
    print("\n1. Checking imports...")
    
    errors = []
    
    # Check core GECS module
    try:
        from src.gec_score import (
            compute_gecs_feature,
            compute_gecs_features_batch,
            gecs_statistical_summary
        )
        print("   ✓ src.gec_score imports successful")
    except Exception as e:
        errors.append(f"   ✗ src.gec_score import failed: {e}")
        print(errors[-1])
    
    # Check pipeline integration
    try:
        from src.run_pipeline import run_pipeline
        print("   ✓ src.run_pipeline imports successful")
    except Exception as e:
        errors.append(f"   ✗ src.run_pipeline import failed: {e}")
        print(errors[-1])
    
    # Check stats analysis
    try:
        from src.stats_analysis import gecs_classification_metrics
        print("   ✓ src.stats_analysis imports successful")
    except Exception as e:
        errors.append(f"   ✗ src.stats_analysis import failed: {e}")
        print(errors[-1])
    
    return len(errors) == 0, errors


def verify_files():
    """Verify all required files exist."""
    print("\n2. Checking file structure...")
    
    required_files = [
        "src/gec_score.py",
        "src/run_pipeline.py",
        "src/stats_analysis.py",
        "run_with_gecs.py",
        "docs/GECS_SUMMARY.md",
        "data/HC3/hc3_sample.json"
    ]
    
    missing = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"   ✓ {file_path}")
        else:
            missing.append(file_path)
            print(f"   ✗ {file_path} - MISSING")
    
    return len(missing) == 0, missing


def verify_api_key():
    """Check if API key is configured."""
    print("\n3. Checking API key configuration...")
    
    try:
        with open("src/gec_score.py", "r") as f:
            content = f.read()
            if "API_KEY = \"sk-proj-" in content and len(content.split("API_KEY")[1].split('"')[1]) > 20:
                print("   ✓ API key configured")
                return True
            else:
                print("   ⚠ API key may not be configured properly")
                return False
    except Exception as e:
        print(f"   ✗ Error checking API key: {e}")
        return False


def verify_dependencies():
    """Check if required packages are installed."""
    print("\n4. Checking dependencies...")
    
    required = {
        'openai': 'OpenAI API client',
        'rouge': 'Rouge scoring',
        'sklearn': 'Scikit-learn (classification metrics)',
        'numpy': 'Numerical operations',
        'pandas': 'Data manipulation',
        'tqdm': 'Progress bars'
    }
    
    missing = []
    for package, description in required.items():
        try:
            __import__(package)
            print(f"   ✓ {package} - {description}")
        except ImportError:
            missing.append(package)
            print(f"   ✗ {package} - {description} - NOT INSTALLED")
    
    if missing:
        print(f"\n   Install missing packages with:")
        print(f"   pip install {' '.join(missing)}")
    
    return len(missing) == 0, missing


def verify_integration():
    """Verify pipeline integration."""
    print("\n5. Checking pipeline integration...")
    
    try:
        import inspect
        from src.run_pipeline import run_pipeline
        
        # Check function signature
        sig = inspect.signature(run_pipeline)
        params = list(sig.parameters.keys())
        
        if 'enable_gecs' in params:
            print("   ✓ enable_gecs parameter present")
        else:
            print("   ✗ enable_gecs parameter missing")
            return False
        
        if 'gecs_model' in params:
            print("   ✓ gecs_model parameter present")
        else:
            print("   ✗ gecs_model parameter missing")
            return False
        
        print("   ✓ Pipeline integration verified")
        return True
        
    except Exception as e:
        print(f"   ✗ Pipeline integration check failed: {e}")
        return False


def main():
    """Run all verification checks."""
    print()
    
    results = {
        "Imports": verify_imports(),
        "Files": verify_files(),
        "API Key": (verify_api_key(), []),
        "Dependencies": verify_dependencies(),
        "Integration": (verify_integration(), [])
    }
    
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for check, (passed, errors) in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check}: {status}")
        if errors:
            for error in errors:
                print(f"  - {error}")
        all_passed = all_passed and passed
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("✅ ALL CHECKS PASSED!")
        print("\nGECS integration is functional and ready to use.")
        print("\nQuick start:")
        print("  python run_with_gecs.py")
        print("\nOr in code:")
        print("  from src.run_pipeline import run_pipeline")
        print("  results = run_pipeline('data.csv', enable_gecs=True)")
        return 0
    else:
        print("⚠️  SOME CHECKS FAILED")
        print("\nPlease address the issues above before using GECS.")
        print("See docs/GECS_SUMMARY.md for setup instructions.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
