import cv2
import os
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataframe_generator import *

standard_size = (256, 256)  # Tamanho padrão para redimensionar as imagens

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

def save_patches(patches, positions, output_dir, prefix, original_shape, patch_size=(32, 32)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    shape_filename = os.path.join(output_dir, f"{prefix}_shape.txt")
    with open(shape_filename, 'w') as f:
        f.write(f"{original_shape[0]},{original_shape[1]},{original_shape[2]}\n")
        f.write(f"Number of Patches: {len(patches)}\n")
        f.write(f"Patch Size: {patch_size[0]},{patch_size[1]}\n")
    for idx, (patch, position) in enumerate(zip(patches, positions)):
        patch_filename = os.path.join(output_dir, f"{prefix}_patch_{position[0]}_{position[1]}.png")
        cv2.imwrite(patch_filename, patch)

def resize_image(image, size):
    return cv2.resize(image, size)

def create_images_set(X_train, X_test, y_train, y_test, patch_size=(32, 32), output_dir_train='patches_train', output_dir_test='patches_test', standard_size=(256, 256)):
    # Processar imagens de treino
    for path, category in zip(X_train, y_train):
        image = cv2.imread(path)
        if image is not None:
            idx = os.path.splitext(os.path.basename(path))[0]  # Usar o nome da imagem como índice
            resized_image = resize_image(image, standard_size)
            image_patches, positions = create_patches(resized_image, patch_size)
            category_dir = os.path.join(output_dir_train, str(category), f'image_{idx}')
            save_patches(image_patches, positions, category_dir, f'image_{idx}', resized_image.shape, patch_size)

    # Processar imagens de teste
    for path, category in zip(X_test, y_test):
        image = cv2.imread(path)
        if image is not None:
            idx = os.path.splitext(os.path.basename(path))[0]  # Usar o nome da imagem como índice
            resized_image = resize_image(image, standard_size)
            image_patches, positions = create_patches(resized_image, patch_size)
            category_dir = os.path.join(output_dir_test, str(category), f'image_{idx}')
            save_patches(image_patches, positions, category_dir, f'image_{idx}', resized_image.shape, patch_size)

def load_patches(directory, patch_size=(32, 32)):
    patches = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.png'):
                patch = cv2.imread(os.path.join(root, filename), cv2.IMREAD_GRAYSCALE)
                if patch is not None and patch.shape == patch_size:
                    patches.append(patch.flatten())
    return np.array(patches)
