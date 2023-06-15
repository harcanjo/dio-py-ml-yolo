import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor

# Classe personalizada para o conjunto de dados contendo gatos e cachorros
class CustomDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.dataset = CocoDetection(root_dir, root_dir, transform=ToTensor())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        # Filtrar apenas as anotações das classes "gato" (categoria 17) e "cachorro" (categoria 18)
        filtered_target = [obj for obj in target if obj["category_id"] in [17, 18]]
        return image, filtered_target

# Diretório onde estão as imagens e os arquivos de anotações
data_dir = "caminho/para/os/dados"

# Criar o conjunto de dados personalizado
custom_dataset = CustomDataset(data_dir)

# Carregar os dados em lotes usando DataLoader
batch_size = 8
dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

# Treinamento do modelo
for images, targets in dataloader:
    # Aqui você treina seu modelo YOLO com as imagens e as anotações dos gatos e cachorros
    # Certifique-se de adaptar essa parte para a biblioteca/framework YOLO específica que você está utilizando

    # Exemplo de impressão das classes encontradas em cada lote
    for target in targets:
        class_ids = [obj["category_id"] for obj in target]
        print("Classes encontradas:", class_ids)

    # Mais código de treinamento aqui...
