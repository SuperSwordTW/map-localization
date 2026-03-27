import numpy as np
from PIL import Image
import os

def pano_to_cubemap(image_path, output_dir="output"):
    # Load the panorama
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    img_array = np.array(img)

    # The size of each cube face is usually 1/4 of the panorama width
    face_size = width // 4
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    faces = ['front', 'back', 'top', 'bottom']
    
    for i,face in enumerate(faces):
        print(f"Generating {face} face...")
        # Create a grid for the face
        out_img = np.zeros((face_size, face_size, 3), dtype=np.uint8)
        
        # Vectorized coordinate mapping
        i, j = np.indices((face_size, face_size))
        
        # Map 2D face coordinates to 3D unit cube coordinates
        x, y, z = 0, 0, 0
        
        # Normalize coordinates to [-1, 1]
        a = 2.0 * i / face_size - 1.0
        b = 2.0 * j / face_size - 1.0

        if face == 'front':  x, y, z =  1.0,  a,  b
        elif face == 'back':   x, y, z = -1.0,  a, -b
        elif face == 'left':   x, y, z =  b,   a,  1.0
        elif face == 'right':  x, y, z = -b,   a, -1.0
        elif face == 'top':    x, y, z =  a,   1.0, -b
        elif face == 'bottom': x, y, z = -a,  -1.0, -b

        # Convert 3D to Spherical (phi, theta)
        theta = np.arctan2(y, x)
        r = np.sqrt(x*x + y*y)
        phi = np.arctan2(z, r)

        # Map Spherical to Panorama coordinates (u, v)
        u = (theta + np.pi) / (2 * np.pi) * width
        v = (np.pi/2 - phi) / np.pi * height

        # Bounds checking and pixel sampling
        u = np.clip(u.astype(int), 0, width - 1)
        v = np.clip(v.astype(int), 0, height - 1)

        out_img[i, j] = img_array[v, u]
        
        # Save the face
        Image.fromarray(out_img).save(os.path.join(output_dir, f"{face}.jpg"))

    print("Done! Faces saved in:", output_dir)

# Usage
pano_to_cubemap("photos/1.jpg")