import os
import argparse
import random
import math

import numpy as np
import csv
from PIL import Image

import torch
torch.set_num_threads(6)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import wandb
from models import ResNet, BasicBlock

class CustomImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Return image, label (dummy for this example), and file name
        label = -1  
        return image, label, self.image_files[idx]


def set_seed(seed=42):
    """
    Set random seeds for all possible random processes.
    :param seed: int
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sort_file_names(file_names):
        # file names are "image_1.png2, "image_2.pnf", ..., "image_99.png"
        # sort them in ascending order
        sorted_file_names = sorted(file_names, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        return sorted_file_names


def run(model, train_loaders, val_loaders, test_loaders, wandb, args, device):

    t_a_p, t_a_n, tt_acc = [], [], []
        
    for idx, train_data in enumerate(train_loaders):
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # lists to keep track of losses and accuracies
        train_loss, train_acc = [], []

        for epoch in range(args.epochs):
            print(f"[INFO]: Epoch {epoch+1} of {args.epochs}")

            train_epoch_loss, train_epoch_acc = train(model, train_data, 
                                                optimizer, criterion, device)
            
            valid_epoch_loss_p, valid_epoch_acc_p, valid_epoch_loss_n, valid_epoch_acc_n = validate(model, val_loaders[idx][0], val_loaders[idx][1],  
                                                            criterion, device)
            
            train_loss.append(train_epoch_loss)
            train_acc.append(train_epoch_acc)

            valid_epoch_loss = (valid_epoch_loss_n + valid_epoch_loss_p)/2
            valid_epoch_acc = (valid_epoch_acc_p + valid_epoch_acc_n)/2

            eval_dict = {f"train_epoch_loss_t{idx}": train_epoch_loss, f"train_epoch_acc_t{idx}": train_epoch_acc,
                         f"valid_epoch_loss_t{idx}": valid_epoch_loss, f"valid_epoch_acc_t{idx}": valid_epoch_acc}

            if wandb:
                wandb.log(eval_dict)


        t_acc, t_acc_p, t_acc_n = test(model, test_loaders, device)

        t_a_p = t_a_p + t_acc_p 
        t_a_n = t_a_n + t_acc_n 
        tt_acc = tt_acc + t_acc

    row = len(train_loaders)
    col_t = len(test_loaders) 

    t_acc_arr_p = np.asarray(t_a_p)
    test_acc_p = t_acc_arr_p.reshape(row, col_t)
    t_acc_arr_n = np.asarray(t_a_n)
    test_acc_n = t_acc_arr_n.reshape(row, col_t)
    t_acc_arr = np.asarray(tt_acc)
    test_acc = t_acc_arr.reshape(row, col_t)

    return  test_acc_p, test_acc_n, test_acc


def train(model, trainloader, optimizer, criterion, device=None):

    model.train()
    print('Training')

    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0

    for _, data in enumerate(trainloader):
        counter += 1

        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        # forward pass
        outputs = model(image)

        # calculate the loss
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()

        # calculate the accuracy
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()

        # backpropagation
        loss.backward()

        # update the optimizer parameters
        optimizer.step()
    
    # loss and accuracy for the complete epoch
    loss = train_running_loss / counter
    acc = 100. * (train_running_correct / len(trainloader.dataset))
    return loss, acc


def validate(model, testloader_p, testloader_n, criterion, device):
    model.eval()

    valid_running_loss_p = 0.0
    valid_running_correct_p = 0

    valid_running_loss_n = 0.0
    valid_running_correct_n = 0

    counter_p = 0
    counter_n = 0

    with torch.no_grad():
        for i, data in enumerate(testloader_p):
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(image)

            # calculate the loss
            loss = criterion(outputs, labels)
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            
            counter_p += 1
            valid_running_loss_p += loss.item()
            valid_running_correct_p += (preds == labels).sum().item()


        for i, data in enumerate(testloader_n):
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(image)

            # calculate the loss
            loss = criterion(outputs, labels)
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)

            counter_n += 1
            valid_running_loss_n += loss.item()
            valid_running_correct_n += (preds == labels).sum().item()
        
    # loss and accuracy for the complete epoch
    epoch_loss_p = valid_running_loss_p / counter_p
    epoch_loss_n = valid_running_loss_n / counter_n

    epoch_acc_p = 100. * (valid_running_correct_p / len(testloader_p.dataset))
    epoch_acc_n = 100. * (valid_running_correct_n / len(testloader_n.dataset))

    return epoch_loss_p, epoch_acc_p, epoch_loss_n, epoch_acc_n


def test(model, test_loader, device):

    # lists used to store accuracy metrics for the model's performance during testing:
    M_a, M_a_p, M_a_n = [], [], []
    model.eval()
    
    for _, (y_p, y_n) in test_loader.items():
        testloader_p = y_p
        testloader_n = y_n
        correct_p = 0
        correct_n = 0

        with torch.no_grad():
            for i, data in enumerate(testloader_p):
                
                image, labels = data
                image = image.to(device)
                labels = labels.to(device)

                output = model(image)
                
                pred = output.argmax(dim=1, keepdim=True)
                correct_p += pred.eq(labels.view_as(pred)).sum().item()

            for i, data in enumerate(testloader_n):
                
                image, labels = data
                image = image.to(device)
                labels = labels.to(device)

                output = model(image)
                
                pred = output.argmax(dim=1, keepdim=True)
                correct_n += pred.eq(labels.view_as(pred)).sum().item()

        acc_p = 100. * correct_p / len(testloader_p.dataset)
        acc_n = 100. * correct_n / len(testloader_n.dataset)
        
        M_a_p.append(acc_p)
        M_a_n.append(acc_n)
        M_a.append((acc_n+acc_p)/2)

    # Return the combined accuracy, positive accuracy, and negative accuracy
    return M_a, M_a_p, M_a_n


def predict(model, test_loader, device):
    model.eval()
    pred_labels = {}
    with torch.no_grad():
        for _, data in enumerate(test_loader):
            image, labels, file_names = data
            image = image.to(device)
            labels = labels.to(device)

            output = model(image)
            pred = output.argmax(dim=1, keepdim=True)
            for i in range(len(file_names)):
                pred_labels[file_names[i]] = pred.cpu().numpy().flatten()[i]
    print(pred_labels)
    return pred_labels


def get_dataset(args):

    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    train_t0 = args.dataset_path + '/train/images/t0'
    train_t1 = args.dataset_path + '/train/images/t1'
    train_t2 = args.dataset_path + '/train/images/t2'

    val_t0 = args.dataset_path + '/val/images/t0'
    val_t1 = args.dataset_path + '/val/images/t1'
    val_t2 = args.dataset_path + '/val/images/t2'

    test_t0 = args.dataset_path + '/test/images/t0'
    test_t1 = args.dataset_path + '/test/images/t1'
    test_t2 = args.dataset_path + '/test/images/t2'

    test_global_path = args.test_path_global

    BATCH_SIZE = args.batch_size
    kwargs = {'num_workers':0, 'pin_memory':False} 
    
    
    test_global_dataset = CustomImageDataset(test_global_path, transform=transformer)
    test_global_loader = DataLoader(dataset=test_global_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
    
    traindataset0 = datasets.ImageFolder(root=train_t0, transform=transformer)
    valdataset0 = datasets.ImageFolder(root=val_t0, transform=transformer)
    positives = [i for i, (x, y) in enumerate(valdataset0) if  y == 1.0]
    negatives = [i for i, (x, y) in enumerate(valdataset0) if  y == 0.0]
    valdataset0_p = torch.utils.data.Subset(valdataset0, positives)
    valdataset0_n = torch.utils.data.Subset(valdataset0, negatives)

    testdataset0 = datasets.ImageFolder(root=test_t0, transform=transformer)
    positives = [i for i, (x, y) in enumerate(testdataset0) if  y == 1.0]
    negatives = [i for i, (x, y) in enumerate(testdataset0) if  y == 0.0]
    testdataset0_p = torch.utils.data.Subset(testdataset0, positives)
    testdataset0_n = torch.utils.data.Subset(testdataset0, negatives)

    traindataset1 = datasets.ImageFolder(root=train_t1, transform=transformer)
    valdataset1 = datasets.ImageFolder(root=val_t1, transform=transformer)
    positives = [i for i, (x, y) in enumerate(valdataset1) if  y == 1.0]
    negatives = [i for i, (x, y) in enumerate(valdataset1) if  y == 0.0]
    valdataset1_p = torch.utils.data.Subset(valdataset1, positives)
    valdataset1_n = torch.utils.data.Subset(valdataset1, negatives)

    testdataset1 = datasets.ImageFolder(root=test_t1, transform=transformer)
    positives = [i for i, (x, y) in enumerate(testdataset1) if  y == 1.0]
    negatives = [i for i, (x, y) in enumerate(testdataset1) if  y == 0.0]
    testdataset1_p = torch.utils.data.Subset(testdataset1, positives)
    testdataset1_n = torch.utils.data.Subset(testdataset1, negatives)

    traindataset2 = datasets.ImageFolder(root=train_t2, transform=transformer)
    valdataset2 = datasets.ImageFolder(root=val_t2, transform=transformer)
    positives = [i for i, (x, y) in enumerate(valdataset2) if  y == 1.0]
    negatives = [i for i, (x, y) in enumerate(valdataset2) if  y == 0.0]
    valdataset2_p = torch.utils.data.Subset(valdataset2, positives)
    valdataset2_n = torch.utils.data.Subset(valdataset2, negatives)

    testdataset2 = datasets.ImageFolder(root=test_t2, transform=transformer)
    positives = [i for i, (x, y) in enumerate(testdataset2) if  y == 1.0]
    negatives = [i for i, (x, y) in enumerate(testdataset2) if  y == 0.0]
    testdataset2_p = torch.utils.data.Subset(testdataset2, positives)
    testdataset2_n = torch.utils.data.Subset(testdataset2, negatives)

    train0_loader = DataLoader(dataset= traindataset0, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    train1_loader = DataLoader(dataset= traindataset1, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    train2_loader = DataLoader(dataset= traindataset2, batch_size=BATCH_SIZE, shuffle=True, **kwargs)

    val0_loader_p = DataLoader(dataset= valdataset0_p, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    val1_loader_p = DataLoader(dataset= valdataset1_p, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    val2_loader_p = DataLoader(dataset= valdataset2_p, batch_size=BATCH_SIZE, shuffle=True, **kwargs)

    val0_loader_n = DataLoader(dataset= valdataset0_n, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    val1_loader_n = DataLoader(dataset= valdataset1_n, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    val2_loader_n = DataLoader(dataset= valdataset2_n, batch_size=BATCH_SIZE, shuffle=True, **kwargs)

    test0_loader_p = DataLoader(dataset= testdataset0_p, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    test1_loader_p = DataLoader(dataset= testdataset1_p, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    test2_loader_p = DataLoader(dataset= testdataset2_p, batch_size=BATCH_SIZE, shuffle=True, **kwargs)

    test0_loader_n = DataLoader(dataset= testdataset0_n, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    test1_loader_n = DataLoader(dataset= testdataset1_n, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    test2_loader_n = DataLoader(dataset= testdataset2_n, batch_size=BATCH_SIZE, shuffle=True, **kwargs)

    train_loaders = [train0_loader, train1_loader, train2_loader]
    val_loaders = {0: [val0_loader_p, val0_loader_n], 1: [val1_loader_p, val1_loader_n], 2:[val2_loader_p, val2_loader_n]}
    test_loaders = {0: [test0_loader_p, test0_loader_n], 1: [test1_loader_p, test1_loader_n], 2:[test2_loader_p, test2_loader_n]}

    return train_loaders, val_loaders, test_loaders, test_global_loader


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Continual Coonfounding Dataset')

    parser.add_argument('--wandb', action='store_true', help='Log run to Weights and Biases.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('-results', '--results_dir', type=str, help='path to store results', required=True)
    parser.add_argument('-name', '--dataset_type', type=str, help='name of dataset variant')
    parser.add_argument('--save_model', action='store_true', help='save model weights', default=True)
    parser.add_argument('-e', '--epochs', type=int, default=50, help='number of epochs to train our network for')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='number of images per batch')
    parser.add_argument('-m', '--model_name', type=str, default='Resnet', help='select the model you would like to run training with')

    parser.add_argument('-path', '--dataset_path', type=str, help='path for dataset for either disjoint or strict')

    parser.add_argument('-gt0', '--test_path_global', type=str, help='path for unconfounded test')


    args = parser.parse_args()

    seed = args.seed
    set_seed(seed)

    out_dir = os.path.dirname(args.results_dir)
    save_model = args.save_model
    dataset_type = args.dataset_type
    model_name = args.model_name
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    if model_name == 'Resnet':
        model = ResNet(img_channels=3, num_layers=18, num_classes=2, block=BasicBlock).to(device)
        model_name = 'Resnet18'
        

    run_id =  wandb.util.generate_id()
    print(f'Sampled new wandb run_id {run_id}.')
    
    wandb.init(project='confounding_continual_dataset', name=os.path.dirname(args.dataset_type),
                id=run_id, resume=True, config=args)
        
    
    print('Loading datasets...')
    train_loaders, val_loaders, test_loaders, test_loaders_global = get_dataset(args)
    test_acc_p, test_acc_n, test_acc = run(model, train_loaders, val_loaders, test_loaders, wandb, args, device)

    pred_labels = predict(model, test_loaders_global, device)


    if save_model:  
        results = args.results_dir
        if not os.path.isdir(f'{results}/{dataset_type}/{model_name}/seed_{seed}'):
            os.makedirs(f'{results}/{dataset_type}/{model_name}/seed_{seed}')
        np.save(f'{results}/{dataset_type}/{model_name}/seed_{seed}/test_acc_p.npy', test_acc_p)
        np.save(f'{results}/{dataset_type}/{model_name}/seed_{seed}/test_acc_n.npy', test_acc_n)
        np.save(f'{results}/{dataset_type}/{model_name}/seed_{seed}/test_acc.npy', test_acc)

        ### upload the csv file
        with open('labels.csv', 'w') as file:
            file.write("ID,TARGET\n")
            images_in_order = sort_file_names(list(pred_labels.keys()))
            for image in images_in_order:
                image_name = image.split('_')[1].split('.')[0]
                file.write(f"image_{image_name},{pred_labels[image]}\n")
    