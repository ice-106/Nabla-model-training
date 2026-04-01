#!/usr/bin/env python3
"""
Script to read and inspect SOKE model output files (.pkl)
Handles NumPy version compatibility issues.
"""

import pickle
import sys

# Import numpy with compatibility handling
try:
    import numpy as np
except ImportError as e:
    print(f"Error: NumPy is not installed: {e}")
    print("\nPlease install numpy:")
    print("  pip install numpy")
    sys.exit(1)


def read_soke_output(filepath):
    """
    Read and display contents of SOKE model output pickle file.
    
    Args:
        filepath: Path to the .pkl file
    
    Returns:
        dict: The loaded data from the pickle file
    """
    try:
        # Load the pickle file with encoding for compatibility
        with open(filepath, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        print("\nIf you're seeing numpy._core errors, try:")
        print("  pip install --upgrade numpy")
        print("Or if using conda:")
        print("  conda update numpy")
        sys.exit(1)
    
    print("="*70)
    print("SOKE Model Output File Inspector")
    print("="*70)
    print(f"\nFile: {filepath}")
    print(f"\nData type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"\nDictionary keys: {list(data.keys())}")
        print("\n" + "-"*70)
        
        for key, value in data.items():
            print(f"\nKey: '{key}'")
            print(f"  Type: {type(value).__name__}")
            
            if isinstance(value, np.ndarray):
                print(f"  Shape: {value.shape}")
                print(f"  Dtype: {value.dtype}")
                print(f"  Value range: [{value.min():.4f}, {value.max():.4f}]")
                print(f"  Mean: {value.mean():.4f}, Std: {value.std():.4f}")
                
                # Check if all zeros
                if np.all(value == 0):
                    print(f"  ⚠️  WARNING: All values are zero!")
                
                # Show number of frames if 2D
                if len(value.shape) == 2:
                    print(f"  → {value.shape[0]} frames × {value.shape[1]} features")
                    
            elif isinstance(value, str):
                print(f"  Value: {value}")
                
            elif isinstance(value, (list, tuple)):
                print(f"  Length: {len(value)}")
                if len(value) > 0:
                    print(f"  First element type: {type(value[0]).__name__}")
            else:
                print(f"  Value: {value}")
    
    else:
        print(f"\nData: {data}")
    
    print("\n" + "="*70)
    
    return data


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python read_soke_output.py <path_to_pkl_file>")
        print("\nExample:")
        print("  python read_soke_output.py S007397_P0007_T00.pkl")
        sys.exit(1)
    
    filepath = sys.argv[1]
    data = read_soke_output(filepath)
    
    # Optional: Return the data for further processing
    # You can modify this script to save or process the data as needed