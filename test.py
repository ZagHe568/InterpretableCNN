import os, sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report    # 生成混淆矩阵函数
from models import *
from cifar10 import test_loader


def plot_confusion_matrix(confustion_mat, labels_name, title):
    confustion_mat = confustion_mat.astype('float') / confustion_mat.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(confustion_mat, interpolation='nearest')    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def test(model, testloader, device, muted_class):
    model_preds = []
    true_labels = []

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, muted=muted_class)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            true_labels.extend(targets.to('cpu').numpy())
            model_preds.extend(predicted.to('cpu').numpy())

    acc = 100.*correct/total
    return acc, model_preds, true_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Test')
    parser.add_argument('--batchsize', default=128, type=int, help='batch size')
    parser.add_argument('--muted_classes', default='', type=str, help='classes to be muted split by comma')
    args = parser.parse_args()
    print(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = resnet18()
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    model.to(device)
    model_path = './checkpoint/ckpt.t7'
    checkpoint = torch.load(model_path)
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)

    print('Successfully Load Model: ', os.path.basename(model_path))

    testloader = test_loader(args.batchsize)
    muted_classes = [int(i)-1 for i in args.muted_classes.split(',')] if args.muted_classes != '' else None
    acc, model_preds, true_labels = test(model, testloader, device, muted_classes)

    labels_name = ['plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    print(classification_report(model_preds, true_labels, target_names=labels_name))

    confustion_mat = confusion_matrix(model_preds, true_labels)
    plot_confusion_matrix(confustion_mat, labels_name, "HAR Confusion Matrix")
    plt.savefig('./HAR_cm.png', format='png')
    print(confustion_mat)