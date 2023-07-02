import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision import models

import os
from PIL import Image
import argparse



class Ensemble(nn.Module):

    def __init__(self, modelA, modelB, input):
        super(Ensemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB

        self.fc1 = nn.Linear(input, 10)

    def forward(self, x):
        out1 = self.modelA(x)
        prob1 = torch.softmax(out1, dim=1)
        out2 = self.modelB(x)
        prob2 = torch.softmax(out2, dim=1)

        prob =( prob1 + prob2 ) /2

        return prob
class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes = os.listdir(root)
        self.class_to_idx = {c: int(c) for i, c in enumerate(self.classes)}
        self.imgs = []
        for c in self.classes:
            class_dir = os.path.join(root, c)
            for filename in os.listdir(class_dir):
                path = os.path.join(class_dir, filename)
                self.imgs.append((path, self.class_to_idx[c])) 
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target




class CustomDataset_Nolabel(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        ImageList = os.listdir(root)
        self.imgs = []
        for filename in ImageList:
            path = os.path.join(root, filename)
            self.imgs.append(path) 
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        path = self.imgs[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img



####################
#If you want to use your own custom model
#Write your code here
####################

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

####################
#Modify your code here
####################
def model_selection(selection):
    if selection == "resnet":
        model = models.resnet18(weights='IMAGENET1K_V1')
        model.conv1 =  nn.Conv2d(3, 64, kernel_size=3,stride=1, padding=1, bias=False)
        model.layer4 = Identity()
        model.fc = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),
            nn.Linear(256, 10)
        )
    elif selection == "vgg":
        model = models.vgg11_bn(weights='IMAGENET1K_V1')
        model.features = nn.Sequential(*list(model.features.children())[:-7])
        model.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.8),
            nn.Linear(in_features=25088, out_features=10, bias=True)
        )
    elif selection == "mobilenet":
        model = models.mobilenet_v2(weights='IMAGENET1K_V2')
        model.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=1280, out_features=10, bias=True)
        )


    return model


def cotrain(net1,net2, labeled_loader, unlabled_loader, optimizer1_1, optimizer1_2, optimizer2_1, optimizer2_2, criterion):
    #The inputs are as below.
    #{First model, Second model, Loader for labeled dataset with labels, Loader for unlabeled dataset that comes without any labels, 
    net1.train()
    net2.train()
    train_loss1 = 0
    train_loss2 = 0
    num_samples = 0
    correct1 = 0
    correct2 = 0
    total = 0
    k = 0.8
    #labeled_training
    for batch_idx, (inputs, targets) in enumerate(labeled_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer1_1.zero_grad()
        optimizer2_1.zero_grad()

        outputs1 = net1(inputs)
        outputs2 = net2(inputs)

        loss1 = criterion(outputs1, targets)
        train_loss1 += loss1.item() * inputs.size(0)
        loss2 = criterion(outputs2, targets)
        train_loss2 += loss2.item() * inputs.size(0)

        num_samples += inputs.size(0)

        total += targets.size(0)
        _, predicted1 = outputs1.max(1)
        correct1 += predicted1.eq(targets).sum().item()

        _, predicted2 = outputs2.max(1)
        correct2 += predicted2.eq(targets).sum().item()

        loss1.backward()
        optimizer1_1.step()

        loss2.backward()
        optimizer2_1.step()
    avg_loss1 = train_loss1 / num_samples
    avg_loss2 = train_loss2 / num_samples
    res1 = 100. * correct1 / total
    res2 = 100. * correct2 / total
    print("labeled train loss: {:.4f}, {:.4f}".format(avg_loss1, avg_loss2))
    print("labeled train res: {}, {}".format(res1, res2))
####################
#Add your code here
####################

    #unlabeled_training    
    for batch_idx, inputs in enumerate(unlabled_loader):
        inputs = inputs.cuda()

        optimizer1_2.zero_grad()
        optimizer2_2.zero_grad()

        outputs1 = net1(inputs)
        predictions1 = torch.argmax(outputs1, dim=1)

        outputs2 = net2(inputs)
        predictions2 = torch.argmax(outputs2, dim=1)

        confidence1 = torch.max(outputs1, dim=1).values
        confidence2 = torch.max(outputs2, dim=1).values

        confident_mask = (predictions1 == predictions2) & (confidence1 > k) & (confidence2 > k)
        confident_inputs = inputs[confident_mask]
        pseudo_outputs1 = net1(confident_inputs)
        pseudo_outputs2 = net2(confident_inputs)
        confident_labels = predictions1[confident_mask]
        pseudo_loss1 = criterion(pseudo_outputs1, confident_labels)
        pseudo_loss1.backward()
        optimizer1_2.step()
        pseudo_loss2 = criterion(pseudo_outputs2, confident_labels)
        pseudo_loss2.backward()
        optimizer2_2.step()

        agreed_predictions = torch.where(predictions1 == predictions2, predictions1, -1)
        confident_predictions = torch.where(confidence1 > k, predictions1, -1) | torch.where(
            confidence2 > k, predictions2, -1)


####################
#Add your code here
####################
        
        #hint : 
        #agree = predicted1 == predicted2
        #pseudo_labels from agree
    



def test(net, testloader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        return 100. * correct / total









if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--test',  type=str,  default='False')
    parser.add_argument('--student_abs_path',  type=str,  default='./')
    args = parser.parse_args()


    batch_size =  16
    if args.test == 'False':
        train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(64, scale=(0.2, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
        test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
        
        dataset = CustomDataset(root = './data/Semi-Supervised_Learning/labeled', transform = train_transform)
        labeled_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        dataset = CustomDataset_Nolabel(root = './data/Semi-Supervised_Learning/unlabeled', transform = train_transform)
        unlabeled_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
        dataset = CustomDataset(root = './data/Semi-Supervised_Learning/val', transform = test_transform)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    else :
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
    if not os.path.exists(os.path.join(args.student_abs_path, 'logs', 'Semi-Supervised_Learning')):
        os.makedirs(os.path.join(args.student_abs_path, 'logs', 'Semi-Supervised_Learning'))

    
    
    model_sel_1 =  'resnet'
    model_sel_2 =  'mobilenet'


    model1 = model_selection(model_sel_1)
    model2 = model_selection(model_sel_2)
    
    params_1 = sum(p.numel() for p in model1.parameters() if p.requires_grad) / 1e6
    params_2 = sum(p.numel() for p in model2.parameters() if p.requires_grad) / 1e6

    if torch.cuda.is_available():
        model1 = model1.cuda()
    if torch.cuda.is_available():
        model2 = model2.cuda()
        
    #You may want to write a loader code that loads the model state to continue the learning process
    #Since this learning process may take a while.
    
    
    if torch.cuda.is_available():
        criterion = nn.CrossEntropyLoss().cuda()
    else :
        criterion = nn.CrossEntropyLoss()    
        
    
    optimizer1_1 = optim.Adam(model1.parameters(), lr=0.0002, weight_decay=0.00007)
    optimizer2_1 = optim.Adam(model2.parameters(), lr=0.0002, weight_decay=0.0001)

    optimizer1_2 = optim.Adam(model1.parameters(), lr=0.00001)
    optimizer2_2 = optim.Adam(model2.parameters(), lr=0.00001)

    epoch = 20

    if args.test == 'False':
        assert params_1 < 7.0, "Exceed the limit on the number of model_1 parameters" 
        assert params_2 < 7.0, "Exceed the limit on the number of model_2 parameters" 

        best_result_1 = 0
        best_result_2 = 0
        ensemble = Ensemble(model1, model2, 10).cuda()
        for e in range(0, epoch):
            cotrain(model1, model2, labeled_loader, unlabeled_loader, optimizer1_1, optimizer1_2, optimizer2_1, optimizer2_2, criterion)
            tmp_res_1 = test(ensemble, val_loader)
            # You can change the saving strategy, but you can't change file name/path for each model
            print ("[{}th epoch, model_1] ACC : {}".format(e, tmp_res_1))
            if best_result_1 < tmp_res_1:
                best_result_1 = tmp_res_1
                torch.save(model1.state_dict(),  os.path.join('./logs', 'Semi-Supervised_Learning', 'best_model_1.pt'))

            tmp_res_2 = test(model2, val_loader)
            # You can change save strategy, but you can't change file name/path for each model
            print ("[{}th epoch, model_2] ACC : {}".format(e, tmp_res_2))
            if best_result_2 < tmp_res_2:
                best_result_2 = tmp_res_2
                torch.save(model2.state_dict(),  os.path.join('./logs', 'Semi-Supervised_Learning', 'best_model_2.pt'))
        print('Final performance {} - {}  // {} - {}', best_result_1, params_1, best_result_2, params_2)

            
    else:
        dataset = CustomDataset(root = '/data/23_1_ML_challenge/Semi-Supervised_Learning/test', transform = test_transform)
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        model1.load_state_dict(torch.load(os.path.join(args.student_abs_path, 'logs', 'Semi-Supervised_Learning', 'best_model_1.pt'), map_location=torch.device('cuda')))
        res1 = test(model1, test_loader)
        
        model2.load_state_dict(torch.load(os.path.join(args.student_abs_path, 'logs', 'Semi-Supervised_Learning', 'best_model_2.pt'), map_location=torch.device('cuda')))
        res2 = test(model2, test_loader)
        
        if res1>res2:
            best_res = res1
            best_params = params_1
        else :
            best_res = res2
            best_params = params_2
            
        print(best_res, ' - ', best_params)        