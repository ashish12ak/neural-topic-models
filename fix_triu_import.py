#!/usr/bin/env python

"""
Fix for the gensim.matutils import error.
This script patches the matutils.py file to import triu from numpy instead of scipy.linalg
"""

import os
import sys
import re

def patch_matutils_file():
    """
    Find gensim's matutils.py file and patch it to use numpy.triu instead of scipy.linalg.triu
    """
    # First, try to find the matutils.py file
    try:
        import gensim
        gensim_path = os.path.dirname(gensim.__file__)
        matutils_path = os.path.join(gensim_path, 'matutils.py')
        
        if not os.path.exists(matutils_path):
            print(f"Error: Could not find matutils.py at {matutils_path}")
            return False
        
        print(f"Found matutils.py at {matutils_path}")
        
        # Read the file
        with open(matutils_path, 'r') as f:
            content = f.read()
        
        # Check if it imports triu from scipy.linalg
        if 'from scipy.linalg import triu' in content:
            # Replace with numpy import
            modified_content = content.replace('from scipy.linalg import triu', 'from numpy import triu')
            
            # Create backup
            backup_path = matutils_path + '.bak'
            with open(backup_path, 'w') as f:
                f.write(content)
            
            # Write modified file
            with open(matutils_path, 'w') as f:
                f.write(modified_content)
            
            print(f"Successfully patched {matutils_path}")
            print(f"Backup saved to {backup_path}")
            return True
        else:
            print("triu import not found in matutils.py or already patched")
            return False
        
    except ImportError:
        print("Error: Could not import gensim. Make sure it's installed.")
        return False

if __name__ == "__main__":
    print("Fixing gensim matutils.py triu import...")
    success = patch_matutils_file()
    
    if success:
        print("Patch completed successfully!")
    else:
        print("Patch failed or was not needed.")
    
    sys.exit(0 if success else 1) 