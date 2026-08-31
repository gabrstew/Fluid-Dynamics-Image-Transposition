import subprocess
import sys

def install_dependencies():
    dependencies = [
        'taichi',
        'numpy',
        'PyQt5',
        'jax',
        'jaxlib',
        'Pillow',
        'opencv-python'
    ]
    
    print("Installing required dependencies...")
    for package in dependencies:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError:
            print(f"Error installing {package}")
            continue
        print(f"Successfully installed {package}")

if __name__ == "__main__":
    install_dependencies()
    print("\nAll dependencies installation completed.")
    print("You can now run the fluid morphing program by running fluid_morph.py")