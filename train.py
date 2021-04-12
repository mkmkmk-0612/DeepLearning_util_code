import torch
from torchvision import transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import argparse
import torch.nn as nn
import torchvision.models as models
from models import ResNet
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    #model = models.resnet101(pretrained=True).cuda()
    model = ResNet.resnet101().cuda()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, default='pdchest_data/')
parser.add_argument('--batch', type=int, default=8)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

args = parser.parse_args()

# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss().cuda()

optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay, nesterov=True)
#optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

train_transform = transforms.Compose([transforms.Resize(512),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])

valid_transform = transforms.Compose([transforms.Resize(512),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])

#test_transform = transforms.Compose([transforms.Resize(512),
#                                        transforms.ToTensor(),
#                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#                                    ])

train_dataset = ImageFolder(args.path+'train', train_transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch, shuffle=True)

valid_dataset = ImageFolder(args.path+'valid', valid_transform)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch, shuffle=True)

#test_dataset = ImageFolder(args.path+'test', test_transform)
#test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch, shuffle=False)

# Strating Train
for e in range(args.epochs):
    print('Epoch {}/{}'.format(e, args.epochs - 1))

    train_tq = tqdm(train_loader, total=int(len(train_loader)))
    model.train()

    for i, (images, labels) in enumerate(train_tq):
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)

        optimizer.zero_grad()

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    validation_loss = 0
    valid_tq = tqdm(valid_loader, total=int(len(valid_loader)))

    val_acc = 0
    min_loss = 0
    c = 1

    # Starting Validation
    with torch.no_grad():
        for j, (images, labels) in enumerate(valid_tq):
            model.eval()
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)

            val_acc += accuracy(outputs, labels)

            loss = criterion(outputs, labels)
            validation_loss += loss.detach().cpu().numpy()
            c += 1

        print('Accuracy : {}'.format(val_acc / c))
        print('----------Validation loss    :   {}----------'.format(validation_loss / c))

        if min_loss < validation_loss/(len(valid_loader)*args.batch):
            min_loss = validation_loss/(len(valid_loader)*args.batch)
            torch.save(model.state_dict(), './weights/model_' + str(e) + '.ckpt')
            print('{} epoch save model ...'.format(e))
'''
# Starting Test
with torch.no_grad():
    model = models.resnet101(pretrained=True).cuda()
    model.load_state_dict(torch.load('weights/model.ckpt'))
    model.eval()
    test_acc = 0
    t = 0

    test_tq = tqdm(test_loader, total=int(len(test_loader)))

    for k, (images, labels) in enumerate(test_tq):
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)

        t += 1
        test_acc += accuracy(outputs, labels)

    print('Test Accuracy : {}'.format(test_acc / t))
'''
