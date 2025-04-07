"""
Neural Topic Models - Text Analysis Package
"""

# Try to import the actual module
try:
    from .run_20newsgroups import main
except ImportError as e:
    # If import fails, provide a helpful error message
    def main():
        print("ERROR: Failed to import text analysis module. Details:")
        print(f"  {str(e)}")
        print("\nPlease ensure all dependencies are installed:")
        print("  pip install torch-sparse torch-geometric")
        print("Or use simplified_main.py which doesn't require these dependencies.")

__all__ = ['main'] 