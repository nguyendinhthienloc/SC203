#!/usr/bin/env python
"""
Clean up redundant GECS files after refactoring.

This script safely removes files that have been consolidated or moved to src/.
"""

import os
from pathlib import Path

# Files to delete (redundant after refactoring)
FILES_TO_DELETE = [
    "gecs.py",                              # → src/gec_score.py
    "gecs_demo.py",                         # → run_with_gecs.py
    "run_gecs_analysis.py",                 # → run_with_gecs.py
    "scripts/convert_csv_to_json_gecs.py",  # One-time use script
    "scripts/install_gecs_deps.py",         # Redundant
    "GECS_COMPLETE.md",                     # → docs/GECS_SUMMARY.md
    "GECS_README.md",                       # → docs/GECS_SUMMARY.md
    "GECS_CHECKLIST.md",                    # → docs/GECS_SUMMARY.md
    "docs/GECS_INTEGRATION_SUMMARY.md",     # → docs/GECS_SUMMARY.md
    "docs/GECS_INTEGRATION.md",             # → docs/GECS_SUMMARY.md
    "docs/GECS_ANALYSIS.md",                # → docs/GECS_SUMMARY.md
    "install_gecs.sh",                      # Redundant
    "cleanup_gecs.sh",                      # Bash version (this Python version is cleaner)
]

def main():
    print("=" * 60)
    print("GECS Refactoring Cleanup")
    print("=" * 60)
    print("\nThis will delete redundant files after GECS refactoring.")
    print("Files have been consolidated into:")
    print("  • src/gec_score.py (core module)")
    print("  • run_with_gecs.py (runner script)")
    print("  • docs/GECS_SUMMARY.md (documentation)")
    print("\n" + "=" * 60)
    
    # Check which files exist
    existing = []
    missing = []
    
    for file_path in FILES_TO_DELETE:
        if Path(file_path).exists():
            existing.append(file_path)
        else:
            missing.append(file_path)
    
    print(f"\nFiles to delete: {len(existing)}")
    for f in existing:
        print(f"  • {f}")
    
    if missing:
        print(f"\nAlready deleted/missing: {len(missing)}")
        for f in missing:
            print(f"  • {f}")
    
    if not existing:
        print("\n✅ All files already cleaned up!")
        return
    
    print("\n" + "=" * 60)
    response = input("Proceed with deletion? (yes/no): ").strip().lower()
    
    if response == 'yes':
        print("\nDeleting files...")
        deleted = 0
        failed = []
        
        for file_path in existing:
            try:
                os.remove(file_path)
                print(f"  ✓ Deleted {file_path}")
                deleted += 1
            except Exception as e:
                print(f"  ✗ Failed to delete {file_path}: {e}")
                failed.append((file_path, str(e)))
        
        print("\n" + "=" * 60)
        print(f"✅ Successfully deleted {deleted} files")
        
        if failed:
            print(f"\n⚠️  Failed to delete {len(failed)} files:")
            for path, error in failed:
                print(f"  • {path}: {error}")
        
        print("\nRefactoring complete! GECS is now organized:")
        print("  📁 src/gec_score.py - Core GECS module")
        print("  📁 run_with_gecs.py - Interactive runner")
        print("  📁 docs/GECS_SUMMARY.md - Complete documentation")
    else:
        print("\nCancelled. No files were deleted.")


if __name__ == "__main__":
    main()
