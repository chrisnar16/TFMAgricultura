import os
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder

class CherryTreeDataset(Dataset):
    def __init__(self, root_dir, transform=None, formats = ('RGB.JPG', 'RED.TIF','GRE.TIF','NIR.TIF','REG.TIF'), concatenate=True, balance=True, healthy_ratio=4):
        self.root_dir = root_dir
        self.transform = transform
        self.formats = formats
        self.samples = []
        self.concatenate = concatenate
        self.label_map = {'Healthy': 0, 'Disease': 1}
        self.counts = {'Healthy': 0, 'Disease': 0}
        self.temp_samples = {'Healthy': [], 'Disease': []}

        labels = []
        # Recorrer todas las subcarpetas y recopilar rutas de imagen
        for subdir, dirs, files in os.walk(root_dir):
            images = {}
            for file in files:
                if file.endswith(formats):  # Añade aquí otros tipos si son necesarios
                    path = os.path.join(subdir, file)
                    group_key = file.split('_')[0]  # Suponemos que el prefijo define la imagen única
                    if group_key not in images:
                        images[group_key] = []
                    images[group_key].append(path)

            # Asegurarse de que cada grupo tiene todos los espectros necesarios
            for key, paths in images.items():
                if len(paths) == len(formats):
                    sorted_paths = sorted(paths, key=lambda x: self.formats.index(x.split('_')[-1]))
                    label = subdir.split(os.sep)[-2]  # Cambio de índice aquí para seleccionar "Armillaria_Stage_1"
                    if(label != 'Healthy'):
                        label = 'Disease'
                    self.counts[label] += 1
                    self.temp_samples[label].append((sorted_paths, self.label_map[label]))

        if balance:
            print('here')
            # Determinar cuántas muestras de "Healthy" seleccionar
            num_disease = len(self.temp_samples['Disease'])
            num_healthy = min(len(self.temp_samples['Healthy']), num_disease * healthy_ratio)
            random.shuffle(self.temp_samples['Healthy'])  # Mezclar antes de seleccionar
            self.counts['Healthy'] = num_healthy
            self.samples = self.temp_samples['Disease'] + self.temp_samples['Healthy'][:num_healthy]
            random.shuffle(self.samples)  # Mezclar el dataset final para entrenamiento
        else:
            self.samples = self.temp_samples['Healthy'] + self.temp_samples['Disease']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        #print(idx)
        paths, label = self.samples[idx]
        image_tensors = []
        for path in paths:
            image = Image.open(path)
            if 'TIF' in path:
                if image.mode == 'RGB':
                    image = image.convert('L')
                elif image.mode == 'I;16':
                    image = np.array(image, dtype=np.float32) / 65535
                    image = Image.fromarray(np.uint8(image * 255))
            # Aplica transformaciones definidas en el constructor\
                image = self.transform(image)
                normalize = transforms.Normalize(mean=[0.45], std=[0.225])
                normalized_image = normalize(image)
            else:
                image = self.transform(image)
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                normalized_image = normalize(image)
            image_tensors.append(normalized_image)
        if self.concatenate:
            image_tensor = torch.cat(image_tensors, dim=0)  # Concatena a lo largo del eje del canal
            return image_tensor, label
        else:
            return image_tensors, label
    def print_class_counts(self):
        for label, count in self.counts.items():
            print(f"Number of samples for {label}: {count}")
