import os
import numpy as np
import cv2

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

def load_patches_by_category(base_dir, categories):
    patches_by_category = {}
    
    for category in categories:
        category_patches = {}
        category_dir = os.path.join(base_dir, str(category))
        
        for root, _, files in os.walk(category_dir):
            files = [f for f in files if f.endswith('.png') and '_patch_' in f]
            files = sorted(files, key=lambda x: (int(x.split('_')[1]), int(x.split('_')[3]), int(x.split('_')[4].split('.')[0])))

            for filename in files:
                try:
                    parts = filename.split('_')
                    image_id = int(parts[1])
                    y = int(parts[3])
                    x = int(parts[4].split('.')[0])
                    patch = cv2.imread(os.path.join(root, filename), cv2.IMREAD_GRAYSCALE)
                    if patch is not None:
                        if image_id not in category_patches:
                            category_patches[image_id] = ([], [])
                        category_patches[image_id][0].append(patch.flatten())
                        category_patches[image_id][1].append((y, x))
                except (IndexError, ValueError) as e:
                    print(f"Error processing file {filename}: {e}")
                    continue

        patches_by_category[category] = category_patches
    
    return patches_by_category
