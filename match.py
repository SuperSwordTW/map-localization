import h5py
from pathlib import Path

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

def main():
    matches_file = "project/outputs/live_frame.jpg_matches.h5"
    query_img = "project/inputs/queries/frame_live_frame.jpg"
    
    # Run Search
    node_id, score = identify_nearest_node(query_img, matches_file)

    if node_id:
        node_name = Path(node_id).name.split('_')[0]
        print(f"\nSUCCESS! Nearest Node: {node_name} (Source: {node_id}) with {score} inliers.")
    else:
        print("\nFAILURE: No matches found.")

if __name__ == "__main__":
    main()