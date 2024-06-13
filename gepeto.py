import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from skimage.util import view_as_windows
from skimage.transform import resize
from skimage.io import imread
import os
import zipfile
import shutil
import random

# Função para extrair arquivos zip
def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Função para separar imagens em subpastas
def separate_images_into_subfolders(data_dir):
    # Paths to the subfolders
    cats_folder = os.path.join(data_dir, 'cats')
    dogs_folder = os.path.join(data_dir, 'dogs')

    # Create subfolders if they don't exist
    if not os.path.exists(cats_folder):
        os.makedirs(cats_folder)
    if not os.path.exists(dogs_folder):
        os.makedirs(dogs_folder)

    # List all files in the data directory
    for filename in os.listdir(data_dir):
        # Check if the file is an image
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            src_path = os.path.join(data_dir, filename)
            if 'cat' in filename:
                dest_path = os.path.join(cats_folder, filename)
            elif 'dog' in filename:
                dest_path = os.path.join(dogs_folder, filename)
            else:
                continue
            
            # Move the file to the appropriate subfolder
            shutil.move(src_path, dest_path)

# Função para carregar e preprocessar imagens
def load_and_preprocess_images(image_dir, target_size=(128, 128), sample_fraction=0.25):
    images = []
    filenames = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]
    sample_size = int(len(filenames) * sample_fraction)
    sampled_filenames = random.sample(filenames, sample_size)
    
    for filename in sampled_filenames:
        img_path = os.path.join(image_dir, filename)
        img = imread(img_path)
        img_resized = resize(img, target_size, anti_aliasing=True)
        images.append(img_resized)
    return np.array(images)

# Função para dividir imagens em patches
def extract_patches(images, patch_size=(32, 32)):
    patches = []
    for img in images:
        img_patches = view_as_windows(img, patch_size, step=patch_size)
        for i in range(img_patches.shape[0]):
            for j in range(img_patches.shape[1]):
                patches.append(img_patches[i, j].reshape(-1))
    return np.array(patches)

# Função para imputar NaNs
def impute_nans(patches):
    imputer = SimpleImputer(strategy='mean')
    patches = imputer.fit_transform(patches)
    return patches

# Função para normalizar patches
def normalize_patches(patches, mean=None, std=None):
    patches = patches.astype(np.float32)
    if mean is None or std is None:
        mean = np.mean(patches, axis=0)
        std = np.std(patches, axis=0)
    normalized_patches = (patches - mean) / (std + 1e-8)  # Pequeno valor para evitar divisão por zero
    return normalized_patches, mean, std

# Função para calcular a métrica OOD para uma imagem individual
def calculate_ood_for_image(image, pca):
    image = image.reshape(1, -1)  # Garantir que a imagem seja 2D
    projected_data = pca.transform(image)
    reconstructed_data = pca.inverse_transform(projected_data)
    residual = image - reconstructed_data

    norm_residual = np.linalg.norm(residual)
    norm_image = np.linalg.norm(image)

    ood_metric = norm_residual / (norm_image + 1e-8)  # Evitar divisão por zero
    return ood_metric

# Calcular a métrica OOD para cada patch
def calculate_ood_for_patches(patches, pca):
    ood_metrics = []
    for patch in patches:
        ood_metric = calculate_ood_for_image(patch, pca)
        ood_metrics.append(ood_metric)
    return np.array(ood_metrics)

# Aplicar PCA para cada categoria e visualizar componentes
def apply_pca_and_visualize(patches_by_category, n_components=0.99):
    pca_by_category = {}
    num_components_95_dict = {}
    for category, patches in patches_by_category.items():
        if patches.size == 0:
            continue  # Pular se não houver patches para esta categoria

        # Imputar e normalizar patches
        patches = impute_nans(patches)
        patches, mean, std = normalize_patches(patches)

        # Aplicar PCA
        pca = PCA(n_components=n_components)
        pca.fit(patches)
        components = pca.components_
        variance = pca.explained_variance_

        pca_by_category[category] = pca

        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        num_components_95 = np.where(cumulative_variance >= 0.95)[0][0] + 1
        num_components_95_dict[category] = num_components_95

        print("Category: " + category)
        print("Number of components that explain 95% of variance: " + str(num_components_95))

    min_num_components = min(num_components_95_dict.values())
    return pca_by_category, min_num_components

# Função principal
def main():
    # Caminhos para os arquivos zip e diretórios de extração
    train_zip_path = 'train.zip'
    test_zip_path = 'test1.zip'
    extract_to = './extracted_data/'

    # Extrair arquivos zip
    extract_zip(train_zip_path, extract_to)
    extract_zip(test_zip_path, extract_to)

    # Separar imagens em subpastas
    train_dir = os.path.join(extract_to, 'train')
    separate_images_into_subfolders(train_dir)

    # Caminhos para as pastas de treinamento e teste
    train_dogs_dir = os.path.join(train_dir, 'dogs')
    train_cats_dir = os.path.join(train_dir, 'cats')

    # Carregar e preprocessar imagens
    dogs_images = load_and_preprocess_images(train_dogs_dir)
    cats_images = load_and_preprocess_images(train_cats_dir)

    # Extrair patches
    dogs_patches = extract_patches(dogs_images)
    cats_patches = extract_patches(cats_images)

    # Agrupar patches por categoria
    patches_by_category = {
        'Dogs': dogs_patches,
        'Cats': cats_patches
    }

    # Normalizar patches e aplicar PCA
    normalized_patches_by_category = {}
    for category, patches in patches_by_category.items():
        normalized_patches, mean, std = normalize_patches(patches)
        normalized_patches_by_category[category] = normalized_patches
    
    pca_by_category, min_components = apply_pca_and_visualize(normalized_patches_by_category)

    # Selecionar uma categoria alvo para calcular OOD
    target_environment = 'Dogs'
    target_patches_environment = normalized_patches_by_category[target_environment]

    ood_metrics = {}

    for category in normalized_patches_by_category:
        test_patches_environment = normalized_patches_by_category[category]
        test_pca = pca_by_category[category]

        # Calcular a métrica OOD para cada patch do target environment usando PCA do target environment
        ood_metrics[category] = calculate_ood_for_patches(target_patches_environment, test_pca)

        # Visualizar a distribuição dos valores de OOD para target environment usando PCA do target environment
        plt.figure(figsize=(10, 6))
        plt.hist(ood_metrics[category], bins=50, alpha=0.75)
        plt.title(f"Distribuição da métrica OOD para {target_environment} (usando PCA do {category})")
        plt.xlabel("OOD")
        plt.ylabel("Frequência")
        plt.grid(True)
        plt.show()

    # Encontrar a categoria com o menor valor de OOD
    min_ood_category = min(ood_metrics, key=lambda k: np.mean(ood_metrics[k]))
    min_ood_value = np.mean(ood_metrics[min_ood_category])
    
    print(f"Estamos no ambiente {min_ood_category} com OOD: {min_ood_value}")

main()
