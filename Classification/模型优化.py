import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import numpy as np
import time
import copy
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 设置matplotlib字体和负号显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 设置随机种子，确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 配置参数
DATA_DIR = "包装盒_resnet"  # 数据目录
MODEL_SAVE_PATH = "resnet18_model_optimized.pth"  # 模型保存路径
BATCH_SIZE = 8  # 减小批次至8，适应小数据集
NUM_WORKERS = 0  # Windows下使用单进程
NUM_EPOCHS = 100  # 增加训练轮次
LEARNING_RATE = 5e-5  # 降低学习率
WEIGHT_DECAY = 1e-4  # 调整正则化强度
PATIENCE = 10  # 增加早停耐心值
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_data_loaders(data_dir, batch_size, num_workers=0):
    """创建增强版数据加载器（含类别平衡采样）"""
    # 优化数据增强策略（减少破坏性变换）
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),  # 降低翻转概率
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),  # 减弱颜色扰动
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 加载数据集
    image_datasets = {x: ImageFolder(
        os.path.join(data_dir, x),
        transform=data_transforms[x]
    ) for x in ['train', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes
    print(f"数据集大小: {dataset_sizes}")
    print(f"类别名称: {class_names}")

    if len(class_names) == 2:
        # 处理类别不平衡（计算权重）
        train_targets = image_datasets['train'].targets
        class_counts = np.bincount(train_targets)
        class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
        weights = class_weights[train_targets]

        # 创建加权采样器
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )

        dataloaders = {
            'train': DataLoader(
                image_datasets['train'],
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers
            ),
            'test': DataLoader(
                image_datasets['test'],
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )
        }
        print(f"类别分布: {dict(zip(class_names, class_counts))}")
        print(f"类别权重: {class_weights.tolist()}")
    else:
        dataloaders = {x: DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=(x == 'train'),
            num_workers=num_workers
        ) for x in ['train', 'test']}

    return dataloaders, dataset_sizes, class_names


def create_model(num_classes):
    """创建优化的模型（减少冻结层数）"""
    # 使用ResNet18并正确加载权重
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # 仅冻结前4层，保留更多可训练层
    for param in list(model.parameters())[:4]:
        param.requires_grad = False

    # 修改全连接层（减少Dropout强度）
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),  # 降低Dropout率
        nn.Linear(num_ftrs, num_classes)
    )
    return model.to(DEVICE)


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=100, patience=10, device='cpu'):
    """训练模型（增强版早停+梯度裁剪）"""
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    counter = 0
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            progress_bar = tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase]))
            for i, (inputs, labels) in progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        # 梯度裁剪防止爆炸
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                progress_bar.set_description(
                    f"{phase} Loss: {loss.item():.4f} Acc: {torch.sum(preds == labels.data).item() / inputs.size(0):.4f}"
                )

            epoch_loss = running_loss / dataloaders[phase].dataset.__len__()
            epoch_acc = running_corrects.double() / dataloaders[phase].dataset.__len__()

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['test_loss'].append(epoch_loss)
                history['test_acc'].append(epoch_acc.item())
                # 更新学习率
                if scheduler:
                    scheduler.step(epoch_loss)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 保存最佳模型
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, MODEL_SAVE_PATH)
                print(f"模型已保存至 {MODEL_SAVE_PATH}")
                counter = 0
            else:
                counter += 1
                print(f"早停计数器: {counter}/{patience}")

        print()

        # 早停判断
        if counter >= patience:
            print(f"早停触发：连续{patience}个epoch验证准确率未提升")
            break

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best test Acc: {best_acc:4f} at epoch {best_epoch}')

    model.load_state_dict(best_model_wts)
    return model, history


def visualize_training(history):
    """可视化训练过程"""
    plt.figure(figsize=(12, 4))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Test Loss')

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['test_acc'], label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Test Accuracy')

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    print("训练历史已保存至 training_history.png")


def evaluate_model(model, dataloader, device='cpu'):
    """评估模型性能"""
    model.eval()

    class_names = dataloader.dataset.classes
    num_classes = len(class_names)
    confusion_matrix = torch.zeros(num_classes, num_classes)

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)
    class_recall = confusion_matrix.diag() / confusion_matrix.sum(0)
    class_f1 = 2 * class_accuracy * class_recall / (class_accuracy + class_recall)
    overall_accuracy = confusion_matrix.diag().sum() / confusion_matrix.sum()

    print("\n模型评估结果:")
    print(f"总体准确率: {overall_accuracy:.4f}")
    print("\n各类别性能:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}:")
        print(f"  准确率: {class_accuracy[i]:.4f}")
        print(f"  召回率: {class_recall[i]:.4f}")
        print(f"  F1分数: {class_f1[i]:.4f}")
        print()

    print("混淆矩阵:")
    print(confusion_matrix)
    return overall_accuracy, confusion_matrix


def main():
    """主函数"""
    if not os.path.exists(DATA_DIR):
        print(f"错误: 数据集目录 {DATA_DIR} 不存在")
        return

    print("准备数据...")
    dataloaders, dataset_sizes, class_names = create_data_loaders(DATA_DIR, BATCH_SIZE, NUM_WORKERS)

    print("初始化模型...")
    model = create_model(len(class_names))

    # 计算类别权重（处理不平衡）
    if len(class_names) == 2:
        train_dataset = dataloaders['train'].dataset
        class_counts = np.bincount(train_dataset.targets)
        class_weights = torch.tensor(class_counts, dtype=torch.float).to(DEVICE)
        class_weights = class_weights / class_weights.sum()  # 归一化
        class_weights = 1.0 - class_weights  # 生成欠采样权重
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"使用类别权重: {class_weights.tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()

    # 使用AdamW优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # 学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-6
    )

    print("开始训练模型...")
    model, history = train_model(
        model, dataloaders, criterion, optimizer, scheduler,
        num_epochs=NUM_EPOCHS, patience=PATIENCE, device=DEVICE
    )

    visualize_training(history)

    print("评估模型性能...")
    evaluate_model(model, dataloaders['test'], device=DEVICE)

    print(f"模型训练完成，已保存至 {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()