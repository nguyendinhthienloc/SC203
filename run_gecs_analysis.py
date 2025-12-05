#!/usr/bin/env python
"""
Run GECS analysis on HC3 dataset.

This script:
1. Checks/installs dependencies
2. Runs GECS analysis on HC3 sample data
3. Generates and displays results

Usage:
    python run_gecs_analysis.py
"""

import sys
import subprocess
import json
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed."""
    required = ['rouge', 'sklearn']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    return missing

def install_dependencies(packages):
    """Install missing packages."""
    print("\n" + "="*60)
    print("Installing missing dependencies...")
    print("="*60)
    
    for package in packages:
        print(f"\nInstalling {package}...")
        try:
            # Map package names to pip names
            pip_name = 'scikit-learn' if package == 'sklearn' else package
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", pip_name, "-q"
            ])
            print(f"✓ {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")
            return False
    
    return True

def run_gecs_analysis():
    """Run GECS analysis on HC3 sample data."""
    print("\n" + "="*60)
    print("GECS Analysis - HC3 Dataset")
    print("="*60)
    
    # Define paths
    test_data = "data/HC3/hc3_sample.json"
    
    # Check if data exists
    if not Path(test_data).exists():
        print(f"\n✗ Error: Test data not found at {test_data}")
        return False
    
    print(f"\n📂 Test data: {test_data}")
    print(f"🔧 Mode: Mock GEC (no API key required)")
    print(f"📊 Using predefined threshold: 0.924")
    
    # Run GECS analysis
    cmd = [
        sys.executable,
        "gecs_demo.py",
        "--test_data_path", test_data,
        "--threshold",
        "--threshold_value", "0.924",
        "--use_mock",
        "--seed", "2023"
    ]
    
    print(f"\n▶ Running: {' '.join(cmd)}\n")
    
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ GECS analysis failed: {e}")
        return False

def display_results():
    """Display analysis results."""
    results_file = "data/HC3/hc3_sample_results_test.json"
    
    if not Path(results_file).exists():
        print(f"\n⚠ Results file not found: {results_file}")
        return
    
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print(f"\n📊 Classification Metrics:")
    print(f"   ROC AUC:    {results['roc_auc']:.4f}")
    print(f"   Accuracy:   {results['accuracy']:.4f}")
    print(f"   Precision:  {results['precision']:.4f}")
    print(f"   Recall:     {results['recall']:.4f}")
    print(f"   F1 Score:   {results['f1']:.4f}")
    
    print(f"\n🎯 Threshold: {results['optimal_threshold']:.4f}")
    
    print(f"\n📋 Confusion Matrix:")
    cm = results['conf_matrix']
    print(f"   [[TN={cm[0][0]:3d}, FP={cm[0][1]:3d}],")
    print(f"    [FN={cm[1][0]:3d}, TP={cm[1][1]:3d}]]")
    
    print(f"\n✓ Full results saved to: {results_file}")

def display_sample_scores():
    """Display sample Rouge-2 scores from processed data."""
    processed_file = "data/HC3/hc3_sample_processed_train.json"
    
    if not Path(processed_file).exists():
        return
    
    print("\n" + "="*60)
    print("SAMPLE ROUGE-2 SCORES")
    print("="*60)
    
    with open(processed_file, 'r') as f:
        data = json.load(f)
    
    # Show first few samples from each group
    human_samples = [d for d in data if d['label'] == 'human'][:3]
    llm_samples = [d for d in data if d['label'] == 'llm'][:3]
    
    print("\n📝 Human texts (sample):")
    for sample in human_samples:
        score = sample.get('llm_text_rouge2_score', 'N/A')
        print(f"   {sample['id']}: Rouge-2 = {score:.4f}" if isinstance(score, float) else f"   {sample['id']}: Rouge-2 = {score}")
        print(f"      Text: {sample['text'][:80]}...")
    
    print("\n🤖 AI texts (sample):")
    for sample in llm_samples:
        score = sample.get('llm_text_rouge2_score', 'N/A')
        print(f"   {sample['id']}: Rouge-2 = {score:.4f}" if isinstance(score, float) else f"   {sample['id']}: Rouge-2 = {score}")
        print(f"      Text: {sample['text'][:80]}...")
    
    # Calculate statistics
    human_scores = [d['llm_text_rouge2_score'] for d in data if d['label'] == 'human' and d.get('llm_text_rouge2_score') is not None]
    llm_scores = [d['llm_text_rouge2_score'] for d in data if d['label'] == 'llm' and d.get('llm_text_rouge2_score') is not None]
    
    if human_scores and llm_scores:
        import numpy as np
        print(f"\n📈 Score Statistics:")
        print(f"   Human - Mean: {np.mean(human_scores):.4f}, Std: {np.std(human_scores):.4f}")
        print(f"   AI    - Mean: {np.mean(llm_scores):.4f}, Std: {np.std(llm_scores):.4f}")
        print(f"   Difference: {abs(np.mean(human_scores) - np.mean(llm_scores)):.4f}")

def main():
    """Main execution function."""
    print("="*60)
    print("GECS Analysis Setup")
    print("="*60)
    
    # Check dependencies
    print("\n1. Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"   Missing: {', '.join(missing)}")
        success = install_dependencies(missing)
        if not success:
            print("\n✗ Failed to install dependencies")
            return 1
    else:
        print("   ✓ All dependencies installed")
    
    # Run analysis
    print("\n2. Running GECS analysis...")
    success = run_gecs_analysis()
    
    if not success:
        print("\n✗ Analysis failed")
        return 1
    
    # Display results
    print("\n3. Displaying results...")
    display_results()
    display_sample_scores()
    
    print("\n" + "="*60)
    print("✅ GECS Analysis Complete!")
    print("="*60)
    print("\n💡 Interpretation:")
    print("   - Rouge-2 F-score measures bigram overlap between original and GEC text")
    print("   - Higher scores = more similar (fewer corrections needed)")
    print("   - AI texts typically have higher Rouge-2 scores (already grammatical)")
    print("   - Human texts may have lower scores (more informal/errors)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
