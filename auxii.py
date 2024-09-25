import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

def plot_spectra(images, labels):
    """
    Función para graficar los tensores de imagen retornados por el Dataset.

    Args:
    - images (list of torch.Tensor): Lista de tensores de imágenes.
    - labels (str): La etiqueta asociada a estas imágenes.
    """
    num_images = len(images)
    fig, axs = plt.subplots(1, num_images, figsize=(num_images * 3, 3))
    if num_images == 1:
        axs = [axs]  # Convertir a lista si solo hay un espectro

    for i, image_tensor in enumerate(images):
        image = F.to_pil_image(image_tensor)  # Convierte el tensor de PyTorch a imagen PIL
        axs[i].imshow(image)
        axs[i].set_title(f'{labels} - Espectro {i+1}')
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()