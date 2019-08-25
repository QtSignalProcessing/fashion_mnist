import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import matplotlib.pyplot as plt
import numpy as np
import timeit
import argparse
import os

import model



def train(model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)  
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            
    print('\nTraining Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    return 100. * correct / len(train_loader.dataset)



def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return 100. * correct / len(test_loader.dataset)


def run(args):
    # Create model
    model_dict = {'toy':model.Net1(),'resnet14':model.resnet14(True), 'resnet14s':model.resnet14()}
    net = model_dict[args.model]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    
    #Data preprocessing
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,),(0.3081,))])
    transform_train = transform
    if args.data_aug:
        transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))])
    
    
    #prepare fashion mnist dataset
    train_dataset = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform_train, download=True)
    
    trainloader = torch.utils.data.DataLoader(train_dataset, 
                                              batch_size=args.batch_size, 
                                              shuffle=True,
                                              num_workers=args.nworkers)

    test_set = torchvision.datasets.FashionMNIST(root='./data', download=False, train=False,transform=transform)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=args.nworkers)
    
    
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)
    
    torch.manual_seed(args.seed)
    start = timeit.default_timer()
    model_path = os.path.join(args.model_path, 'fashion_mnist_' + args.model +'.pt')
    
    
    tar = 0        
    acc_train = 0
    ep = 0
    for epoch in range(1, args.nepochs):
        acc_tr = train(net, device, trainloader, optimizer, epoch)
        acc = test(net, device, testloader)
        #save the model with best accuracy
        if tar < acc:
            tar = acc
            acc_train = acc_tr
            ep = epoch
            
            torch.save(net.state_dict(), model_path)

    stop = timeit.default_timer()

    #print('Time: ', (stop - start)/args.nepochs  
    print('\n Best test accuracy: {:.4f}, training accuracy: {:.4f}, number of epoch {}, training time per epoch: {:.4f}s\n'.format(tar,acc_train, ep,(stop - start)/args.nepochs))
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='toy', help="toy, resnet14, resnet14s")
    parser.add_argument("--lr", type=int, default=0.001, help="learning rate")
    parser.add_argument("--data_aug", type=bool, default=False, help="data augmentation")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--nepochs", type=int, default=50, help="max epochs")
    parser.add_argument("--nworkers", type=int, default=4, help="number of workers")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--model_path", type=str, default='./model', help="Directory to save model")
    args = parser.parse_args()
    run(args)