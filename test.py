import torch
import numpy as np
from hloc import extractors
from hloc.utils.base_model import dynamic_load

def check_netvlad_mean():
    # 1. Load the NetVLAD model with default configuration
    # (This matches the default behavior in hloc extraction scripts)
    conf = {'model_name': 'VGG16-NetVLAD-Pitts30K'}
    Model = dynamic_load(extractors, 'netvlad')
    model = Model(conf).eval()
    
    # 2. Access the preprocessing dictionary
    # The 'mean' is usually loaded from the .mat weights file
    mean_val = model.preprocess['mean']
    std_val = model.preprocess['std']
    
    print(f"--- NetVLAD Preprocessing Params ---")
    print(f"Model Name: {model.conf['model_name']}")
    print(f"Internal Mean (RGB): {mean_val}")
    print(f"Internal Std  (RGB): {std_val}")
    
    # 3. Validation Check
    # Common ImageNet mean is [0.485, 0.456, 0.406]
    # Pitts30k mean is often different.
    print(f"\nIs this standard ImageNet mean? {np.allclose(mean_val, [0.485, 0.456, 0.406], atol=1e-3)}")

if __name__ == "__main__":
    check_netvlad_mean()