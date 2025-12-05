"""
Install additional dependencies for GECS script.
"""
import subprocess
import sys

def install_packages():
    """Install required packages for GECS script."""
    packages = [
        "rouge",
        "openai",
        "torch",
        "scikit-learn"
    ]
    
    print("Installing GECS dependencies...")
    for package in packages:
        print(f"\nInstalling {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")
            return False
    
    print("\n✓ All dependencies installed successfully!")
    return True

if __name__ == "__main__":
    success = install_packages()
    sys.exit(0 if success else 1)
