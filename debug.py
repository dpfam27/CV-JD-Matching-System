import sys
import traceback
from pathlib import Path
import os

try:
    print("=== Script Starting ===")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Try to access the directory
    pdf_dir = Path("data/raw/cv/test").absolute()
    print(f"\nTrying to access: {pdf_dir}")
    print(f"Directory exists? {pdf_dir.exists()}")
    
    # Try to list files
    if pdf_dir.exists():
        files = list(pdf_dir.glob("*.pdf"))
        print(f"\nFound {len(files)} PDF files:")
        for f in files:
            print(f"- {f.name}")
    
    print("\n=== Script Completed Successfully ===")

except Exception as e:
    print("\n!!! ERROR OCCURRED !!!")
    print(f"Error type: {type(e)}")
    print(f"Error message: {str(e)}")
    print("\nFull traceback:")
    traceback.print_exc()
    
finally:
    print("\n=== Script Finished ===")

input("Press Enter to exit...")