from hloc import extract_features, pairs_from_retrieval, match_features
from pathlib import Path

def main():
    project_root = Path('project')
    inputs = project_root / 'inputs'
    outputs = project_root / 'outputs'
    outputs.mkdir(parents=True, exist_ok=True)

    feature_path = outputs / 'features.h5'
    retrieval_path = outputs / 'retrieval.h5'
    pairs_path = outputs / 'pairs-query-retrieval.txt'
    match_path = outputs / 'matches.h5'

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

    # Generate Pairs (NetVLAD)
    print("Generating pairs...")
    pairs_from_retrieval.main(
        retrieval_path, 
        pairs_path, 
        num_matched=5, 
        db_prefix='mapping', 
        query_prefix='queries'
    )

    # Extract Local Features (SuperPoint)
    print("Extracting local features...")
    extract_features.main(
        feature_conf, 
        inputs, 
        feature_path=feature_path,
    )

    # Match Features (SuperGlue)
    print("Matching features...")
    match_features.main(
        matcher_conf, 
        pairs_path, 
        features=feature_path, 
        matches=match_path,
        overwrite=True 
    )

    print(f"Pipeline complete. Matches saved to {match_path}")

if __name__ == "__main__":
    main()