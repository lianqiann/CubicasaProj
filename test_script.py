import os
import numpy as np
import torch
from PIL import Image
import cv2

from house import House
import copy
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import transforms as T
from engine import train_one_epoch, evaluate
import utils
import matplotlib.pyplot as plt
import argparse
from floortrans.loaders import FloorplanSVG, DictToTensor, Compose, RotateNTurns


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
        elif num_obj == 0:
            rand_inds = []
            print('No objects in this image, folder:', self.imgs[idx])
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



def prepare_model(dataset_name, device):


    # our dataset has two classes only - background and person
    num_classes = 1+10
    # use our dataset and defined transformations
    dataset = CubicasaDataset('data/cubicasa5k', dataset_name,get_transform(train=False))

    

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, 
            collate_fn=utils.collate_fn)

    
    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    

    return model, data_loader


def main():
    parser = argparse.ArgumentParser(
        description='MaskRCNN for Cubicasa Dataset')
    

    parser.add_argument('--epoch', type=int, default=None,
                        help="which epoch result model to run, default: None")
    parser.add_argument('--data_name', type=str, default='val',
                        help="which dataset to test, default: val")
    parser.add_argument('--run_all', type=bool, default=False,
                        help="whether to run all models")
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model, data_loader = prepare_model(args.data_name, device)

    if args.epoch:
        model.load_state_dict(torch.load(f'models/maskrcnn_{args.epoch}.pt',map_location='cpu'))
        evaluate(model, data_loader, device=device)
    
    if args.run_all:
        for epoch in range(4):
            model.load_state_dict(torch.load(f'models/maskrcnn_{epoch}.pt',map_location='cpu'))
            evaluate(model, data_loader, device=device)



main()


    

