from hloc import extract_features, pairs_from_retrieval, match_features
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Security, status, Request
from fastapi.security import APIKeyHeader
import os
import shutil
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import h5py

import cv2
import numpy as np
import torch
from hloc.extractors.superpoint import SuperPoint
from hloc.extractors.netvlad import NetVLAD
from hloc.matchers.superglue import SuperGlue
from hloc.utils.base_model import dynamic_load
from PIL import Image, ImageOps
import io
from torchvision import transforms

app = FastAPI()

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

API_KEY = ""
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    # Standard 'Bearer' format check
    if api_key == f"Bearer {API_KEY}":
        return api_key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://map-rho-blue.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "ngrok-skip-browser-warning"],
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on device: {device}")

DB_FEATURES_PATH = Path('project/outputs/db_features.h5')
DB_GLOBAL_PATH = Path('project/outputs/db_retrieval.h5')

conf_sp = {'nms_radius': 4, 'keypoint_threshold': 0.005, 'max_keypoints': 4096}
model_sp = SuperPoint(conf_sp).eval().to(device)

conf_nv = {'model_name': 'VGG16-NetVLAD-Pitts30K'} # Standard HLoc config
model_nv = NetVLAD(conf_nv).eval().to(device)

conf_sg = {'weights': 'outdoor', 'sinkhorn_iterations': 20, 'match_threshold': 0.2}
model_sg = SuperGlue(conf_sg).eval().to(device)

print("Loading database descriptors into RAM...")
db_globals = {}
with h5py.File(DB_GLOBAL_PATH, 'r') as f:
    for name in f.keys():
        db_globals[name] = f[name]['global_descriptor'][()]

db_names = list(db_globals.keys())
db_desc_matrix = np.array([db_globals[n] for n in db_names]) # Shape: (N, 4096)
db_desc_tensor = torch.from_numpy(db_desc_matrix).to(device) # Move to GPU for speed

print(f"Database loaded: {len(db_names)} images.")

def preprocess_netvlad(img_rgb, resize_max=1024): 
    
    # 1. Resize Logic
    h, w = img_rgb.shape[:2]
    scale = resize_max / max(h, w)
    
    if scale < 1.0:
        new_size = (int(round(w * scale)), int(round(h * scale)))
        img_rgb = cv2.resize(img_rgb, new_size, interpolation=cv2.INTER_AREA)

    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
    img_tensor /= 255.0
    
    return img_tensor[None]

def preprocess_superpoint(img_rgb, resize_max=1024):
    # 1. Convert RGB to Grayscale
    # (Previous code used BGR2GRAY, which yields different gray values for RGB input)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # 2. Resize Logic
    h, w = img_gray.shape
    scale = resize_max / max(h, w)
    
    if scale < 1.0:
        new_size = (int(round(w * scale)), int(round(h * scale)))
        img_gray = cv2.resize(img_gray, new_size, interpolation=cv2.INTER_AREA)
        
    # 3. Normalize to [0, 1]
    img_tensor = torch.from_numpy(img_gray / 255.0).float()
    
    # Returns (1, 1, H, W)
    return img_tensor[None, None]

UPLOAD_DIR = "project/inputs/queries"
os.makedirs(UPLOAD_DIR, exist_ok=True)

project_root = Path('project')
inputs = project_root / 'inputs/queries'
inputs.mkdir(parents=True, exist_ok=True)
outputs = project_root / 'outputs'
outputs.mkdir(parents=True, exist_ok=True)

def load_image_from_upload_rgb(file: UploadFile):
    file.file.seek(0)
    file_bytes = file.file.read()
    
    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img = ImageOps.exif_transpose(pil_img) 
    except Exception as e:
        raise ValueError(f"Could not open image: {e}")

    # PIL is RGB by default.
    img_rgb = np.array(pil_img)

    # Handle Alpha channel or Grayscale
    if len(img_rgb.shape) == 2:  # Grayscale
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    elif img_rgb.shape[2] == 4:  # RGBA
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2RGB)
    
    return img_rgb # Return RGB directly


@app.post("/process")
@limiter.limit("5/minute")
def process_image(
    request: Request,
    image: UploadFile = File(...),
    authenticated: str = Depends(verify_api_key)
    ):
    img_rgb = load_image_from_upload_rgb(image)

    with torch.inference_mode():
        img_tensor_sp = preprocess_superpoint(img_rgb).to(device)
        img_tensor_nv = preprocess_netvlad(img_rgb).to(device)

        # Run SuperPoint
        pred_sp = model_sp({'image': img_tensor_sp})
        # Run NetVLAD
        pred_nv = model_nv({'image': img_tensor_nv})

        query_desc = pred_nv['global_descriptor']
        query_desc = query_desc.to(db_desc_tensor.dtype)
        query_desc = query_desc / query_desc.norm(dim=1, keepdim=True)

        scores = torch.matmul(query_desc, db_desc_tensor.t())
        top_k_scores, top_k_indices = torch.topk(scores, k=10)

        best_best_db_name = None
        best_inlier_count = 0

        with h5py.File(DB_FEATURES_PATH, 'r') as f_db:
            for i in range(10):
                best_db_name = db_names[top_k_indices[0, i].item()]

                print(f"{i} match: {best_db_name} (Score: {top_k_scores[0, i].item():.2f})")

                
                db_grp = f_db[best_db_name]
                db_keypoints = torch.from_numpy(db_grp['keypoints'][()]).float().to(device)
                db_descriptors = torch.from_numpy(db_grp['descriptors'][()]).float().to(device)
                db_scores = torch.from_numpy(db_grp['scores'][()]).float().to(device)

                current_dtype = pred_sp['keypoints'][0].dtype

                sg_input = {
                    'image0': img_tensor_sp, # Query image (dummy, SG only needs shapes mostly)
                    'keypoints0': pred_sp['keypoints'][0][None].to(current_dtype),
                    'scores0': pred_sp['scores'][0][None].to(current_dtype),
                    'descriptors0': pred_sp['descriptors'][0][None].to(current_dtype),
                    
                    'image1': img_tensor_sp, # Placeholder
                    'keypoints1': db_keypoints[None].to(current_dtype),
                    'scores1': db_scores[None].to(current_dtype),
                    'descriptors1': db_descriptors[None].to(current_dtype),
                }

                pred_sg = model_sg(sg_input)

                matches = pred_sg['matches0'][0] # Shape (N,)
                valid_matches = matches > -1
                inlier_count = valid_matches.sum().item()

                print(f"Inliers with {best_db_name}: {inlier_count}")

                if inlier_count > 80 and inlier_count > best_inlier_count:
                    best_best_db_name = best_db_name
                    best_inlier_count = inlier_count

    if best_best_db_name != None:
        return {
            "status": "success",
            "message": "Frame processed successfully",
            "nearest_node": f"{best_best_db_name}",
            "inliers": f"{best_inlier_count}"
        }

    return {
        "status": "failure",
        "message": "No matches found or insufficient inliers",
        "nearest_node": None,
        "inliers": f"{best_inlier_count}"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
