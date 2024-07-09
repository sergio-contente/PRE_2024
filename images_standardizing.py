import cv2
import os
import numpy as np
import sys

standard_size = (256, 256)  # Tamanho padrão para redimensionar as imagens

# Função para redimensionar a imagem
def resize_image(image, size):
    return cv2.resize(image, size)

# Função para salvar imagens
def save_image(image, output_dir, base_name):
    os.makedirs(output_dir, exist_ok=True)
    image_name = f"{base_name}.png"
    cv2.imwrite(os.path.join(output_dir, image_name), image)

# Função para criar conjunto de imagens redimensionadas
def create_images_set(X_train, X_test, y_train, y_test, output_dir_train='images_train', output_dir_test='images_test', standard_size=(224, 224)):
    for path, category in zip(X_train, y_train):
        image = cv2.imread(path)
        if image is not None:
            idx = os.path.splitext(os.path.basename(path))[0]  # Usar o nome da imagem como índice
            resized_image = resize_image(image, standard_size)
            category_dir = os.path.join(output_dir_train, str(category))
            save_image(resized_image, category_dir, f'image_{idx}')

    for path, category in zip(X_test, y_test):
        image = cv2.imread(path)
        if image is not None:
            idx = os.path.splitext(os.path.basename(path))[0]  # Usar o nome da imagem como índice
            resized_image = resize_image(image, standard_size)
            category_dir = os.path.join(output_dir_test, str(category))
            save_image(resized_image, category_dir, f'image_{idx}')

def load_images_by_category(base_dir, categories, image_size=(224, 224)):
    images_by_category = {}
    for category in categories:
        category_images = []
        category_dir = os.path.join(base_dir, str(category))
        for root, _, files in os.walk(category_dir):
            for filename in files:
                if filename.endswith('.png'):
                    image_path = os.path.join(root, filename)
                    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Ensure image is read in color
                    if image is not None:
                        resized_image = resize_image(image, image_size)
                        category_images.append(resized_image)
                    else:
                        print(f"Failed to load image: {image_path}")
        images_by_category[category] = np.array(category_images)
    return images_by_category
