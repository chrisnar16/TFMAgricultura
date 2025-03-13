import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from PIL import Image, TiffTags
import os
import numpy as np
import torch

#PATH = '/mnt/d/Maestria/tfm/7144071'
PATH = '/mnt/d/M/TFMAgricultura/7144071'

def plot_spectra(images, label):
    """
    Visualiza las bandas espectrales de una imagen multiespectral.
    
    Args:
        images: tensor de PyTorch con las bandas concatenadas [C, H, W]
        label: etiqueta de la imagen (0: sano, 1: enfermo)
    """
    # Determinar cuántas bandas hay según el número de canales
    num_channels = images.shape[0]
    # Convertir tensor a numpy y normalizar para visualización
    images_np = images.numpy()
    # Crear figura
    fig, axes = plt.subplots(1, num_channels, figsize=(15, 4))
    # Si solo hay un canal, axes no será una matriz
    if num_channels == 1:
        axes = [axes]
    # Nombres de las bandas según el orden que se usó en CherryTreeDataset
    # Esto depende del orden en que se concatenaron las bandas en tu dataset
    band_names = ['RGB_R', 'RGB_G', 'RGB_B', 'NIR', 'REG', 'RED', 'GRE']
    # Asegurarse de que tenemos los nombres correctos
    band_names = band_names[:num_channels]
    
    # Visualizar cada canal
    for i in range(num_channels):
        img = images_np[i]
        
        # Normalizar para visualización
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        # Mostrar imagen
        axes[i].imshow(img, cmap='viridis' if i >= 3 else 'gray')
        axes[i].set_title(f"Banda {band_names[i]}")
        axes[i].axis('off')
    
    status = "Enfermo" if label == 1 else "Sano"
    plt.suptitle(f"Bandas espectrales de un árbol de cerezo ({status})")
    plt.tight_layout()
    plt.show()

def analyze_image(image_path):
    """
    Analiza una imagen y muestra información sobre su profundidad de bits y dimensiones
    
    Args:
        image_path (str): Ruta a la imagen
    """
    # Abrir la imagen
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"Error al abrir la imagen: {e}")
        return

    # Información básica
    print(f"\nAnálisis de la imagen: {os.path.basename(image_path)}")
    print(f"Formato: {img.format}")
    print(f"Modo: {img.mode}")
    print(f"Dimensiones: {img.size}")
    
    # Convertir a array numpy para análisis más detallado
    img_array = np.array(img)
    
    # Análisis por canal
    if len(img_array.shape) > 2:  # Imagen con múltiples canales
        num_channels = img_array.shape[2]
        print(f"\nNúmero de canales: {num_channels}")
        
        for i in range(num_channels):
            channel = img_array[:,:,i]
            bits_needed = int(np.ceil(np.log2(channel.max() + 1)))
            print(f"Canal {i+1}:")
            print(f"  - Bits por pixel: {bits_needed}")
            print(f"  - Valor máximo: {channel.max()}")
            print(f"  - Valor mínimo: {channel.min()}")
    else:  # Imagen en escala de grises
        bits_needed = int(np.ceil(np.log2(img_array.max() + 1)))
        print("\nImagen en escala de grises:")
        print(f"Bits por pixel: {bits_needed}")
        print(f"Valor máximo: {img_array.max()}")
        print(f"Valor mínimo: {img_array.min()}")
    
    # Calcular bits totales por pixel
    dtype_bits = img_array.dtype.itemsize * 8
    total_bits = dtype_bits * (num_channels if len(img_array.shape) > 2 else 1)
    print(f"\nBits totales por pixel: {total_bits}")
    print(f"Tipo de datos: {img_array.dtype}")

def analyze_tiff_metadata(image_path):
    """
    Analiza los metadatos de una imagen TIFF y muestra información sobre su banda espectral
    
    Args:
        image_path (str): Ruta a la imagen TIFF
    """
    try:
        # Verificar que el archivo existe
        if not os.path.exists(image_path):
            print(f"El archivo {image_path} no existe")
            return

        # Abrir la imagen TIFF
        img = Image.open(image_path)
        
        # Verificar que sea un archivo TIFF
        if img.format != 'TIFF':
            print(f"La imagen {image_path} no es un archivo TIFF")
            return
        
        print(f"\nAnálisis de la imagen TIFF: {os.path.basename(image_path)}")
        print(f"Dimensiones: {img.size}")
        print(f"Modo: {img.mode}")
        
        # Obtener todos los metadatos TIFF
        try:
            meta_dict = {
                TiffTags.TAGS[key].name: value
                for key, value in img.tag_v2.items()
            }
        except AttributeError:
            print("No se pudieron obtener los metadatos TIFF")
            meta_dict = {}
        
        # Información específica de la banda
        band_indicators = {
            'wavelength': None,
            'bandwidth': None,
            'band_name': None
        }
        
        # Intentar encontrar información de la banda en los metadatos
        for key, value in meta_dict.items():
            key_lower = key.lower()
            if 'wavelength' in key_lower:
                band_indicators['wavelength'] = value
            elif 'bandwidth' in key_lower:
                band_indicators['bandwidth'] = value
            elif 'band' in key_lower and 'name' in key_lower:
                band_indicators['band_name'] = value
        
        print("\nMetadatos de la banda:")
        for key, value in band_indicators.items():
            if value:
                print(f"{key}: {value}")
        
        # Análisis de bits
        img_array = np.array(img)
        dtype_bits = img_array.dtype.itemsize * 8
        print(f"\nProfundidad de bits: {dtype_bits} bits")
        print(f"Valor máximo: {img_array.max()}")
        print(f"Valor mínimo: {img_array.min()}")
        
        # Imprimir todos los metadatos disponibles
        print("\nTodos los metadatos disponibles:")
        for key, value in meta_dict.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"Error al procesar la imagen: {str(e)}")
    
    finally:
        if 'img' in locals():
            img.close()

def identify_band(wavelength):
    """
    Identifica la banda espectral basada en la longitud de onda
    
    Args:
        wavelength (float): Longitud de onda en nanómetros
    Returns:
        str: Nombre de la banda espectral
    """
    try:
        wavelength = float(wavelength)
        if 450 <= wavelength <= 495:
            return "Azul"
        elif 495 <= wavelength <= 570:
            return "Verde"
        elif 620 <= wavelength <= 750:
            return "Rojo"
        elif 700 <= wavelength <= 730:
            return "Red Edge"
        elif 750 <= wavelength <= 900:
            return "Infrarrojo cercano (NIR)"
        else:
            return "Banda desconocida"
    except (ValueError, TypeError):
        return "Longitud de onda inválida"


def crop_central_region(image, center_ratio=0.6):
    """
    Recorta la región central de la imagen según el ratio proporcionado.
    Funciona con objetos Image de PIL, tensores de PyTorch o arrays de numpy
    
    Args:
        image: Image de PIL, tensor de PyTorch o array de numpy
        center_ratio: proporción del tamaño original (0.6 = 60% del centro)
        
    Returns:
        La imagen recortada en el mismo formato que la entrada
    """
    if isinstance(image, torch.Tensor):
        # Si es un tensor de PyTorch [C, H, W]
        c, h, w = image.shape
    elif isinstance(image, np.ndarray):
        # Si es un array de numpy
        if len(image.shape) == 3:
            h, w, c = image.shape
        else:
            h, w = image.shape
    else:
        # Si es un objeto Image de PIL
        w, h = image.size
    
    # Calcular dimensiones del recorte
    new_h = int(h * center_ratio)
    new_w = int(w * center_ratio)
    
    # Calcular coordenadas de inicio para el recorte
    start_h = (h - new_h) // 2
    start_w = (w - new_w) // 2
    
    # Recortar basado en el tipo de entrada
    if isinstance(image, torch.Tensor):
        return image[:, start_h:start_h+new_h, start_w:start_w+new_w]
    elif isinstance(image, np.ndarray):
        if len(image.shape) == 3:
            return image[start_h:start_h+new_h, start_w:start_w+new_w, :]
        else:
            return image[start_h:start_h+new_h, start_w:start_w+new_w]
    else:
        # Para objetos Image de PIL
        return image.crop((start_w, start_h, start_w + new_w, start_h + new_h))