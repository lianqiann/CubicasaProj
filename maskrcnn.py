import os
import numpy as np
import torch
from PIL import Image

from engine import train_one_epoch, evaluate
import transforms as T
import utils

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from floortrans.loaders.house import House
import copy
import argparse


import cv2

room_classes = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room" ,"Bed Room", "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"]
rooms = ["Outdoor", "Kitchen", "Living Room" ,"Bed Room", "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"]
room_ids = [1,3,4,5,6,7,8,9,10,11]
room_labels = {rooms[i]:i+1 for i in range(len(rooms))}


class CubicasaDataset(object):
    def __init__(self, root, mode, transforms=None):
        self.root = root
        #self.dict = torch.load(f"data/cubicasa5k/instance_info_{mode}.pt")
        self.transforms = transforms
        self.imgs = np.genfromtxt(root + '/'+mode+'.txt', dtype='str')
    
        
    def __getitem__(self, idx):
        # load images ad masks
        
        #instance_info = self.dict[self.imgs[idx]]
        
        # fplan = cv2.imread(self.root + self.imgs[idx]+'F1_original.png')
        # img = cv2.cvtColor(fplan, cv2.COLOR_BGR2RGB)/255.  # correct color channels

        org_img_path = self.root + self.imgs[idx]+'F1_original.png'
        img_path = self.root + self.imgs[idx]+'F1_scaled.png'
        svg_path = self.root + self.imgs[idx]+'model.svg'

        img = Image.open(org_img_path).convert("RGB")

        height, width, _ = cv2.imread(img_path).shape
        height_org, width_org, _ = cv2.imread(org_img_path).shape

        # Getting labels for segmentation and heatmaps
        house = House(svg_path, height, width)
        # Combining them to one numpy tensor
        label = torch.tensor(house.get_segmentation_tensor().astype(np.float32))

        label = label.unsqueeze(0)
        label = torch.nn.functional.interpolate(label,
                                                    size=(height_org, width_org),
                                                    mode='nearest')
        label = label.squeeze(0)[0]


        #############process items##############
        masks = label.data.numpy()
    
        boxes = []
        labels = []
        num_obj = 0
        
        mask_tensor = []
        areas = []

        limit_list = []

        for r in room_ids:
            x = copy.copy(masks)
            x[masks != r] = 0 
            x = x.astype(np.uint8)
            contours, _ = cv2.findContours(x,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            limit_list +=[(r, cot) for cot in contours]
            num_obj+=len(contours)
        
        if num_obj >20:
            rand_inds = np.random.choice(np.arange(num_obj), 20, replace  = False)
        else:
            rand_inds = np.arange(num_obj)
        
        for ind in rand_inds:
            r, tcnt = limit_list[ind]
            im = np.zeros((height,width,3), np.uint8)
            im = cv2.drawContours(im, [tcnt], -1, (255,255,255), -1)
            mask_tensor.append((im[:,:,0]/255).astype(np.int8))
            areas.append(cv2.contourArea(tcnt,False))
            x,y,w,h = cv2.boundingRect(tcnt)
            boxes.append([x,y,x+w,y+h])
            labels.append(room_labels[room_classes[r]])
        
        boxes = torch.FloatTensor(boxes)
        labels = torch.as_tensor(labels, dtype = torch.long)
        areas = torch.FloatTensor(areas)
        
        try:
            mask_tensor = np.stack(mask_tensor, 0)
        except:
            mask_tensor = np.array([])


        #######################################
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = torch.as_tensor(mask_tensor, dtype=torch.uint8)
        target["image_id"] = torch.tensor([idx], dtype = torch.int8)
        target["area"] = areas
        target["iscrowd"] = torch.zeros(num_obj, dtype = torch.int8)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO

    
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model



def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)




def main():
    parser = argparse.ArgumentParser(
        description='Variational Sequential Labelers \
        for Semi-supervised learning')

    parser.add_argument('--epochs', type=int, default=4,
                        help="the number of training epcohs, default: 4")
    parser.add_argument('--train', type=str, default='train',
                        help="the name of training set, default: train")
    parser.add_argument('--val', type=str, default='val',
                        help="the name of training set, default: val")
    parser.add_argument('--test', type=str, default='test',
                        help="the name of test set, default: test")
    parser.add_argument('--batch_size', type=int, default=2,
                        help="batch size, default: 32")
    parser.add_argument('--model_name', type=str, default='maskrcnn',
                        help="model, default: maskrcnn")

    args = parser.parse_args()



    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 1+10
    # use our dataset and defined transformations
    dataset = CubicasaDataset('data/cubicasa5k', args.train,get_transform(train=True))
    dataset_test = CubicasaDataset('data/cubicasa5k', args.test,get_transform(train=False))

    # split the dataset in train and test set
    # indices = torch.randperm(len(dataset)).tolist()
    # dataset = torch.utils.data.Subset(dataset, indices[:-50])
    # dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, 
        collate_fn=utils.collate_fn)
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False, 
        collate_fn=utils.collate_fn)
    
    if args.val!='None':
        dataset_val = CubicasaDataset('data/cubicasa5k', args.val,get_transform(train=False))

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=args.batch_size, shuffle=False, 
            collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = args.epochs

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        if args.val !='None':
            try:
                evaluate(model, data_loader_val, device=device)
            except:
                print('evaluation encouters problem!')


            torch.save(model.state_dict(), f'checkpoints/{args.model_name}_{epoch}.pt')

        print('*'*25+f'epoch {epoch} finished'+'*'*25)

        

    #print("get test results")
    #evaluate(model, data_loader_test, device=device)


    print("That's it!")




main()