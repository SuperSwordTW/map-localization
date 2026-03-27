from hloc import extract_features, pairs_from_retrieval, match_features
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
import os
import shutil
from fastapi.middleware.cors import CORSMiddleware
import h5py
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your site's URL
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "project/inputs/queries"
os.makedirs(UPLOAD_DIR, exist_ok=True)

project_root = Path('project')
inputs = project_root / 'inputs/queries'
inputs.mkdir(parents=True, exist_ok=True)
outputs = project_root / 'outputs'
outputs.mkdir(parents=True, exist_ok=True)

def get_netvlad_score(query_name, db_node_id, query_ret_h5, db_ret_h5):
    with h5py.File(query_ret_h5, 'r') as q_file, h5py.File(db_ret_h5, 'r') as db_file:
        # Get query descriptor
        q_desc = q_file[query_name]['global_descriptor'][()]
        
        # Get database descriptor (hloc uses '-' instead of '/' in some keys)
        db_key = db_node_id.replace('/', '-') 
        if db_key not in db_file:
            db_key = db_node_id # Try original if not found
            
        db_desc = db_file[db_key]['global_descriptor'][()]
        
        # NetVLAD descriptors in hloc are usually L2-normalized, 
        # so dot product = cosine similarity.
        score = np.dot(q_desc, db_desc)
        return float(score)

def identify_nearest_node(query_path, match_h5):
    path_obj = Path(query_path)
    matched_key = f"{path_obj.name}"
    
    best_node = None
    max_inliers = 0
    
    with h5py.File(match_h5, 'r') as f:
        # Find which key style is being used

        # Iterate through the matches in that group
        group = f[matched_key]
        
        for candidate_key in group.keys():
            item = group[candidate_key]
            
            matches = item['matches0'][()]
            
            if matches is not None:
                # Count inliers (indices > -1)
                current_inliers = (matches > -1).sum()
                
                # Un-flatten the candidate name for display
                # e.g., "mapping-68_3.jpg" -> "mapping/68_3.jpg"
                display_name = candidate_key
                if "-" in display_name and "/" not in display_name:
                     parts = display_name.split('-')
                     if len(parts) > 1:
                         display_name = f"{parts[0]}/{'-'.join(parts[1:])}"

                print(f"    Candidate: {display_name} | Inliers: {current_inliers}")
                
                if current_inliers > max_inliers:
                    max_inliers = current_inliers
                    best_node = display_name

    return best_node, max_inliers

@app.post("/process")
def process_image(image: UploadFile = File(...)):
    print(f"Received image: {image.filename}")
    file_path = os.path.join(UPLOAD_DIR, f"frame_{image.filename}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    query_feature_path = outputs / f'{image.filename}_features.h5'
    query_retrieval_path = outputs / f'{image.filename}_retrieval.h5'
    pairs_path = outputs / f'{image.filename}_pairs-query-retrieval.txt'
    match_path = outputs / f'{image.filename}_matches.h5'

    retrieval_conf = extract_features.confs['netvlad']
    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['superglue']
    retrieval_conf['preprocessing']['num_workers'] = 0
    feature_conf['preprocessing']['num_workers'] = 0

    print(f"Processing image: {file_path}")
    # Extract Global Features from query image
    extract_features.main(
        retrieval_conf, 
        inputs, 
        feature_path=query_retrieval_path,
    )

    # Pair query with database images using NetVLAD retrieval
    pairs_from_retrieval.main(
        query_retrieval_path, 
        pairs_path, 
        num_matched=5,
        db_descriptors=Path('project/outputs/db_retrieval.h5')
    )

    # Extract Local Features from query image
    extract_features.main(
        feature_conf, 
        inputs, 
        feature_path=query_feature_path,
    )

    # Match features between query and paired database images
    match_features.main(
        matcher_conf, 
        pairs_path, 
        features=query_feature_path,
        features_ref=Path('project/outputs/db_features.h5'),
        matches=match_path,
        overwrite=True 
    )

    print(f"Finished processing {file_path}. Matches saved to {match_path}")
    
    query_img = "project/inputs/queries/frame_live_frame.jpg"
    matches_file = "project/outputs/live_frame.jpg_matches.h5"

    
    # Run Search
    node_id, score = identify_nearest_node(query_img, matches_file)

    vlad_score = 0.0

    if node_id:
        vlad_score = get_netvlad_score(
            "frame_live_frame.jpg", 
            node_id, 
            query_retrieval_path, 
            Path('project/outputs/db_retrieval.h5')
        )
        node_name = Path(node_id).name.split('_')[0]
        print(f"\nSUCCESS! Nearest Node: {node_name}")
        print(f"Inliers: {score} | NetVLAD Score: {vlad_score:.4f}")
    else:
        print("\nFAILURE: No matches found.")

    # 4. Return data back to the website
    if score > 100:
        return {
            "status": "success",
            "message": "Frame processed successfully",
            "nearest_node": f"{node_id}",
            "inliers": f"{score}"
        }

    return {
        "status": "failure",
        "message": "No matches found or insufficient inliers",
        "nearest_node": None,
        "inliers": f"{score}"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)