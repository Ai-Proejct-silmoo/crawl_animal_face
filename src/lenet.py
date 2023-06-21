import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw

# 동물상 이미지 데이터셋 경로
train_data_dir = './train'
test_data_dir = './test'

classes = ['cat', 'dog', 'rabbit']

# 데이터 전처리
transform = transforms.Compose([
    transforms.Resize((32, 32)),    # 이미지 크기 조정
    transforms.ToTensor(),           # 이미지를 Tensor로 변환
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 이미지 정규화
])

# 데이터 로더 설정
train_dataset = ImageFolder(train_data_dir, transform=transform)
test_dataset = ImageFolder(test_data_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 동물상 분류 모델 정의
class AnimalClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AnimalClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 모델 인스턴스 생성
model = AnimalClassifier(num_classes=3)

# 기기 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 손실 함수 설정
criterion = nn.CrossEntropyLoss()

# 최적화 함수 설정
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 학습 함수
def train_model(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for e in range(epoch):
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            if i % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(e + 1, epoch, i + 1, len(train_loader), loss.item()))

        mean_loss = total_loss / len(train_loader)
        print('Epoch [{}/{}], Mean Loss: {:.4f}'.format(e + 1, epoch, mean_loss))

# 평가 함수
def evaluate_model(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print('Accuracy: {:.2f}%'.format(accuracy))

# 이미지 분류 함수
def classify_image(image_path, model, device, transform):
    model.eval()
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        return classes[predicted.item()]

# 학습 실행
train_model(model, device, train_loader, optimizer, criterion, epoch=30)

evaluate_model(model, device, test_loader)

image_path = '/Users/ohhhgnoeyk/Documents/Folder/crawl_animal_face/src/test/dog/강다니엘15.jpg'  # 분류할 이미지 경로
predicted_label = classify_image(image_path, model, device, transform)
print('Predicted Label:', predicted_label)