"""
Neural Topic Models - Text Generation Package
"""

# Try to import the actual module
try:
    from .run_wikitext import main
except ImportError as e:
    # If import fails, provide a helpful error message
    def main():
        print("ERROR: Failed to import text generation module. Details:")
        print(f"  {str(e)}")
        print("\nPlease ensure all dependencies are installed:")
        print("  pip install torch-sparse torch-geometric")
        print("Or use simplified_main.py which doesn't require these dependencies.")

__all__ = ['main'] 