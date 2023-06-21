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

class LeNet5(nn.Module):
    # torch.nn.Module 클래스: 
    #   모든 신경망 모듈을 위한 base class. 모델을 만들려면 이 클래스를 subclass 해야 함
    #   Module 클래스는 다른 Module 들을 포함할 수 있음
    def __init__(self, batch, n_classes, in_channel, in_width, in_height, is_train=False):
        super().__init__()

        self.batch = batch
        self.n_classes = n_classes
        self.in_channel = in_channel
        self.in_width = in_width
        self.in_height = in_height
        self.is_train = is_train

        ### convolution & pooling layer
        # ※ convolution output = {(W - K + 2P) / S} + 1
        self.conv0 = nn.Conv2d(self.in_channel, 6, kernel_size=5, stride=1, padding=0)
            # torch.nn.Conv2d 클래스: 입력 신호에 대해 2D Convolution을 적용함
            #   필수 파라미터: input_channel, output_channel
            #   보통 입력 크기는 (B, C_in, H, W), 출력 크기는 (B, C_out, H_out, W_out)
            #   자료형으로 TensorFloat32를 사용
        self.pool0 = nn.AvgPool2d(2, stride=2)
            # torch.nn.AvgPool2d 클래스: 입력 신호에 대해 2D average pooling을 적용함
            #   필수 파라미터: kernel_size
        self.conv1 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.AvgPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)

        ### fully-connected layer
        self.fc3 = nn.Linear(120, 84)
            # torch.nn.Linear 클래스: 입력 데이터에 대해 선형 변환을 적용. y = x * A.t + b
            #   필수 파라미터: in_features(각 input sample의 사이즈), out_features
        self.fc4 = nn.Linear(84, self.n_classes)

    def forward(self, x):
        # nn.Module.forward() 메서드: 매 call마다 수행될 내용. subclass에서 재정의되어야 함. 재정의한 부분이 하단
        #   Model 객체를 데이터와 함께 호출하면 자동으로 실행됨. 따라서 my_model = LeNet(input)으로 선언/호출 해도 자동으로 forward 수행됨
        # ※ x의 shape: [B, C, H, W]

        x = self.conv0(x)
        x = torch.tanh(x)
        x = self.pool0(x)
        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = torch.tanh(x)
        
        x = torch.flatten(x, start_dim=1)
            # 4차원을 2차원으로 바꿈 ([b, c, h, w] -> [B, C*H*W])

        x = self.fc3(x)
        x = torch.tanh(x)
        x = self.fc4(x)

        x = x.view(self.batch, -1)
            # Tensor.view() 메서드: 인자로 주어진 tensor의 shape를 변경해 새 tensor로 리턴
            #   파라미터 shape에 -1 입력 시, 다른 dimension으로부터 값을 자동으로 추정함

        x = nn.functional.softmax(x, dim=1)
            # torch.nn.functional.softmax() 함수: [0, 1]의 범위를 갖도록 softmax 함수를 적용

        if self.is_train is False:
            x = torch.argmax(x, dim=1)
                # torch.argmax() 함수: 입력 tensor에 대해 가장 큰 값을 리턴함

        return x


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