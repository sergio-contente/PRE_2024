import cv2
import os
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataframe_generator import *

patch_size = (64, 64)

def create_patches(image, patch_size):
    h, w, c = image.shape
    patches = []
    positions = []

    for i in range(0, h, patch_size[0]):
        for j in range(0, w, patch_size[1]):
            patch = np.zeros((patch_size[0], patch_size[1], c), dtype=image.dtype)
            patch_part = image[i:i + patch_size[0], j:j + patch_size[1]]
            patch[:patch_part.shape[0], :patch_part.shape[1]] = patch_part
            patches.append(patch)
            positions.append((i, j))
    return patches, positions

def save_patches(patches, positions, output_dir, prefix, original_shape):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    shape_filename = os.path.join(output_dir, f"{prefix}_shape.txt")
    with open(shape_filename, 'w') as f:
        f.write(f"{original_shape[0]},{original_shape[1]},{original_shape[2]}")
    for idx, (patch, position) in enumerate(zip(patches, positions)):
        patch_filename = os.path.join(output_dir, f"{prefix}_patch_{position[0]}_{position[1]}.png")
        cv2.imwrite(patch_filename, patch)

def create_images_set(X_train, X_test, patch_size=(64, 64), output_dir_train='patches_train', output_dir_test='patches_test'):
    for path in X_train:
        image = cv2.imread(path)
        if image is not None:
            image_patches, positions = create_patches(image, patch_size)
            save_patches(image_patches, positions, output_dir_train, os.path.splitext(os.path.basename(path))[0], image.shape)

    for path in X_test:
        image = cv2.imread(path)
        if image is not None:
            image_patches, positions = create_patches(image, patch_size)
            save_patches(image_patches, positions, output_dir_test, os.path.splitext(os.path.basename(path))[0], image.shape)

def main():
    df, split_variables = create_dataframe()
    X_train, X_test = split_variables[0], split_variables[1]

    create_images_set(X_train, X_test)

if __name__ == "__main__":
    main()
