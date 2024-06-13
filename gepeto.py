import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Função para imputar NaNs
def impute_nans(patches):
    imputer = SimpleImputer(strategy='mean')
    patches = imputer.fit_transform(patches)
    return patches

# Função para normalizar patches
def normalize_patches(patches):
    patches = patches.astype(np.float32)
    mean = np.mean(patches, axis=0)
    std = np.std(patches, axis=0)
    normalized_patches = (patches - mean) / (std + 1e-8)  # Pequeno valor para evitar divisão por zero
    return normalized_patches

# Função para calcular a métrica OOD para uma imagem individual
def calculate_ood_for_image(image, pca):
    print(f'Forma original da imagem: {image.shape}')
    image = image.reshape(1, -1)  # Garantir que a imagem seja 2D
    print(f'Forma após reshape: {image.shape}')
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
        patches = normalize_patches(patches)

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
def main(patches_by_category):
    pca_by_category, min_components = apply_pca_and_visualize(patches_by_category)
    test_patches_by_category = patches_by_category  # Supondo que test_patches_by_category é o mesmo que patches_by_category

    for category in test_patches_by_category:
        patches = test_patches_by_category[category]
        normalized_patches = normalize_patches(patches)

        # Calcular a métrica OOD para cada patch usando PCA da mesma categoria
        ood_metrics_same_category = calculate_ood_for_patches(normalized_patches, pca_by_category[category])

        # Visualizar a distribuição dos valores de OOD
        plt.figure(figsize=(10, 6))
        plt.hist(ood_metrics_same_category, bins=50, alpha=0.75)
        plt.title(f"Distribuição da métrica OOD para {category} (usando PCA do {category})")
        plt.xlabel("OOD")
        plt.ylabel("Frequência")
        plt.grid(True)
        plt.show()

# Substitua patches_by_category pelo seu dicionário de patches
patches_by_category = {
    'Bedroom': np.random.rand(100, 1024),  # Substitua por seus dados reais
    'Coast': np.random.rand(100, 1024)  # Substitua por seus dados reais
}

main(patches_by_category)
