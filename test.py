import os, sys
import torch
import torch.nn as nn
import numpy as np

import argparse

from sklearn.metrics import confusion_matrix, classification_report    # 生成混淆矩阵函数
import matplotlib.pyplot as plt    # 绘图库

from models import *
from dataLoader import test_loader
import torch.backends.cudnn as cudnn


print('Init Finished!')


def printF(i, total=100):
    i = int( i / total * 100) + 1
    total = 100
    k = i + 1
    str_ = '>'*i + '' ''*(total-k)
    sys.stdout.write('\r'+str_+'[%s%%]'%(i+1))
    sys.stdout.flush()
    if (i >= total -1): print()


def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
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

            true_labels.extend(targets.cpu().numpy())
            model_preds.extend(predicted.cpu().numpy())

    acc = 100.*correct/total
    return acc, model_preds, true_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Test')
    parser.add_argument('--batchsize', default=128, type=int, help='batch size')
    parser.add_argument('muted_classes', default='', type=str, help='classes to be muted')
    args = parser.parse_args()
    print(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = './checkpoint/ckpt.t7'
    model = resnet18(pretrained=True)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    print('Successfully Load Model: ', os.path.basename(model_path))

    testloader = test_loader(args.batchszie)
    muted_class = [int(i) for i in args.muted_classes.split(',')]
    acc, model_preds, true_labels = test(model, testloader, device, muted_class)



    labels_name = ['plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    cm = confusion_matrix(model_preds, true_labels)
    print(cm)
    plot_confusion_matrix(cm, labels_name, "HAR Confusion Matrix")
    plt.savefig('/HAR_cm.png', format='png')
    # plt.show()

    print(classification_report(model_preds, true_labels, target_names=labels_name))
    plot_confusion_matrix(cm, labels_name, "HAR Confusion Matrix")
    plt.savefig('/HAR_cm.png', format='png')
    # plt.show()
    print(classification_report(model_preds, true_labels, target_names=labels_name))