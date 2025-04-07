#!/usr/bin/env python3

"""
Fix for scipy.linalg.triu import issue
"""

import sys
import importlib

def patch_scipy_linalg():
    """Patches scipy.linalg with numpy's triu function"""
    try:
        import scipy.linalg
        if not hasattr(scipy.linalg, 'triu'):
            # Add triu function from numpy to scipy.linalg
            from numpy import triu
            scipy.linalg.triu = triu
            print("Successfully patched scipy.linalg.triu with numpy.triu")
            return True
    except ImportError:
        print("Warning: Could not import scipy.linalg")
    return False

def reload_gensim():
    """Reload gensim module after patching scipy.linalg"""
    try:
        import gensim
        # Force reload gensim module to use the patched scipy.linalg
        importlib.reload(gensim)
        print("Successfully reloaded gensim module")
        return True
    except ImportError:
        print("Warning: Could not import gensim")
    return False

if __name__ == "__main__":
    # Apply the patch
    if patch_scipy_linalg():
        # Reload affected modules
        reload_gensim()
        print("Patch applied successfully")
    else:
        print("Failed to apply patch")
        sys.exit(1) 