import os
import subprocess

def setup_environment():
    """Create project structure and environment"""
    # Create directories
    dirs = ['data/raw', 'data/processed', 'src/data', 'src/models', 'src/api']
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)
    
    # Create __init__.py files
    for dir in ['src/data', 'src/models', 'src/api']:
        with open(f'{dir}/__init__.py', 'w') as f:
            pass

if __name__ == "__main__":
    setup_environment()