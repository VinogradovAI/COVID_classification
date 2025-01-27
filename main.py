import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import resnet18
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
from collections import Counter

# Установка параметров
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 15
IMG_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Классы в датасете
CLASSES = ["COVID", "Normal", "Viral Pneumonia"]

# Функция для загрузки данных
class LungXRayDataset(Dataset):
    def __init__(self, root_dir="data", transform=None):
        """Кастомный Dataset для загрузки рентгеновских снимков."""
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(CLASSES)}

        # Читаем изображения по классам
        for class_name in CLASSES:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(".png"):  # Загружаем только .png файлы
                        img_path = os.path.join(class_dir, img_name)
                        self.data.append((img_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# Применение трансформаций
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Загрузка датасета (теперь корректно передаются аргументы)
dataset = LungXRayDataset(root_dir="data", transform=transform)
print(f"Total images: {len(dataset)}")


# Функция для проверки размерности данных в DataLoader
def check_dataloader(loader, name="Train"):
    images, labels = next(iter(loader))
    print(f"{name} DataLoader check - Images shape: {images.shape}, Labels shape: {labels.shape}")
    assert images.shape[1:] == (3, 224, 224), "Ошибка: Неверная размерность входного тензора изображений!"


# Создание DataLoader
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

check_dataloader(train_loader, "Train")
check_dataloader(val_loader, "Validation")
check_dataloader(test_loader, "Test")

# Определение модели ResNet18
model = resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))  # 3 класса
model = model.to(DEVICE)
print("Model initialized")

# Оптимизатор и функция потерь
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Функция обучения модели
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=EPOCHS):
    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Проверка размерностей на GPU
            assert images.device == DEVICE, "Ошибка: Тензоры изображений не на GPU!"
            assert labels.device == DEVICE, "Ошибка: Тензоры меток не на GPU!"
            assert images.shape[1:] == (3, 224, 224), "Ошибка: Неверная размерность входных изображений!"

            optimizer.zero_grad()
            outputs = model(images)

            # Проверка размерности выхода
            assert outputs.shape == (labels.size(0), len(CLASSES)), f"Ошибка: Неверный выходной тензор {outputs.shape}"

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = 100. * correct / total
        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")

        # Валидация
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        val_acc = 100. * correct / total
        print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Validation Acc: {val_acc:.2f}%")

        # Сохранение лучшей модели
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")

    # Запуск обучения
train_model(model, train_loader, val_loader, criterion, optimizer)

# Оценка модели на тесте
def evaluate_model(model, test_loader):
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Вывод метрик
    report = classification_report(all_labels, all_preds, target_names=CLASSES, digits=4)
    print("Classification Report:\n", report)

    # Вывод матрицы ошибок
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.show()

# Запуск оценки модели
evaluate_model(model, test_loader)