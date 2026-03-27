from hloc import extract_features, pairs_from_retrieval, match_features
from pathlib import Path

def main():
    project_root = Path('project')
    inputs = project_root / 'inputs/mapping'
    outputs = project_root / 'outputs'
    outputs.mkdir(parents=True, exist_ok=True)

    feature_path = outputs / 'db_features.h5'
    retrieval_path = outputs / 'db_retrieval.h5'

    # Configurations
    retrieval_conf = extract_features.confs['netvlad']
    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['superglue']

    retrieval_conf['preprocessing']['num_workers'] = 0
    feature_conf['preprocessing']['num_workers'] = 0

    # Extract Global Features (NetVLAD-based retrieval)
    print("Extracting global features...")
    extract_features.main(
        retrieval_conf, 
        inputs, 
        feature_path=retrieval_path,
    )

    # Extract Local Features (SuperPoint)
    print("Extracting local features...")
    extract_features.main(
        feature_conf, 
        inputs, 
        feature_path=feature_path,
    )

    print(f"Pipeline complete. Global features saved to {retrieval_path} and local features saved to {feature_path}")

if __name__ == "__main__":
    main()