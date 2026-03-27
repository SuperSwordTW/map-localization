import cv2
import os
import py360convert

def process_nodes():
    input_folder = 'photos'
    output_folder = 'inputs/mapping'
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Sampling 8 horizontal views (0 to 315 degrees)
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    v_angles = [0]

    for j in range(1, 146):
        filename = f'{j}.jpg'
        file_path = os.path.join(input_folder, filename)
        
        img = cv2.imread(file_path)
        
        if img is None:
            print(f"⚠️ Warning: Could not find {file_path}")
            continue

        k = 1
        for i, angle in enumerate(angles):
            for v_angle in v_angles:
                crop = py360convert.e2p(
                    e_img=img, 
                    fov_deg=[49.55, 69.39],
                    u_deg=angle if angle <= 180 else angle - 360, 
                    v_deg=v_angle, 
                    out_hw=(640, 480)
                )
                output_filename = f'{j}_{k}.jpg'
                save_path = os.path.join(output_folder, output_filename)
                cv2.imwrite(save_path, crop)
                k += 1
            
        print(f"✅ Node {j:03d} processed successfully.")

if __name__ == "__main__":
    process_nodes()