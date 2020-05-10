import os
import numpy as np
import torch
from PIL import Image
import cv2

from floortrans.loaders.house import House
import copy
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import model_utils.transforms as T
from model_utils.engine import train_one_epoch, evaluate
import model_utils.utils as utils
import matplotlib.pyplot as plt
import argparse
from floortrans.loaders import FloorplanSVG, DictToTensor, Compose, RotateNTurns
from tqdm import tqdm 
from floortrans.metrics import get_evaluation_tensors, runningScore


import warnings



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
   
        org_img_path = self.root + self.imgs[idx]+'F1_original.png'
        img_path = self.root + self.imgs[idx]+'F1_scaled.png'
        svg_path = self.root + self.imgs[idx]+'model.svg'        

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
        masks = cv2.resize(label.data.numpy(), (256,256))
    
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
        
#         if num_obj >20:
#             rand_inds = np.random.choice(np.arange(num_obj), 20, replace  = False)
        if num_obj ==0:
            rand_inds = []
            print('No objects in this image, folder:', self.imgs[idx])
        else:
            rand_inds = np.arange(num_obj)
        
        for ind in rand_inds:
            r, tcnt = limit_list[ind]
            im = np.zeros((256,256,3), np.uint8)
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
        #target["folder"] = torch.tensor([self.imgs[idx]])
        target["area"] = areas
        target["iscrowd"] = torch.zeros(num_obj, dtype = torch.int8)
        
        img = Image.open(org_img_path).convert("RGB")
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class Decode_Maskrcnn(object):
    
    def __init__(self, dataset, idx, model = None, prediction = None, nms = 0.9):

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.folder = dataset.imgs[idx].to(device)
        self.img = dataset[idx][0]
        
        if not prediction:
            model.eval()
            with torch.no_grad():
                self.pred = model([self.img])[0].cpu()
        else:
            self.pred = prediction[0].cpu()
        
        
        self.org_img_path ='./data/cubicasa5k'+self.folder+'F1_original.png'
        self.height_org, self.width_org, _ = cv2.imread(self.org_img_path).shape
        
        self.inds = torchvision.ops.nms(self.pred['boxes'], self.pred['scores'],nms)
        
        
    
    
    ##################GROUND TRUTH####################
    
    def get_groundtruth(self, img_show = True, resize= False):
        
        svg_path ='./data/cubicasa5k'+self.folder+'model.svg'
        img_path ='./data/cubicasa5k'+self.folder+'F1_scaled.png'
        height, width, _ = cv2.imread(img_path).shape
        

        house = House(svg_path, height, width)
        # Combining them to one numpy tensor
        gt = torch.tensor(house.get_segmentation_tensor().astype(np.float32))

        gt = gt.unsqueeze(0)
        gt = torch.nn.functional.interpolate(gt,size=(self.height_org, self.width_org),
                                                            mode='nearest')
        gt = gt.squeeze(0)[0]
        
        if resize:
            gt = cv2.resize(gt.data.numpy(), (256,256)) #return numpy array

        if img_show:
            plt.figure(figsize=(10,10))
            ax = plt.subplot(1, 1, 1)
            plt.title('Ground Truth Rooms and walls', fontsize=20)
            ax.axis('off')
            n_rooms = 12
            rseg = ax.imshow(gt, cmap='rooms', vmin=0, vmax=n_rooms-0.1)
            cbar = plt.colorbar(rseg, ticks=np.arange(n_rooms) + 0.5, fraction=0.046, pad=0.01)
            cbar.ax.set_yticklabels(room_classes, fontsize=20)
            plt.show()
        
        
            
        
        return gt
    
    
    ##################PREDICTION#######################
    
    
    #segmentation map
    def get_segmap(self, thres = 0.1, img_show = True, resize= False):
        
        num,_, H,W =self.pred['masks'].shape

        seg_map = torch.zeros((H,W))
        labels = set()
        for i in self.inds:
            label = int(self.pred['labels'][i].data.numpy())
            if label>=2:
                label+=1
            if label in labels:
                continue
            mask = self.pred['masks'][i,0,:,:]
            seg_map[mask>thres] = label
            

            labels.add(label)
            
        # edges = torch.as_tensor(self.getBordered(seg_map.data.numpy(), 4))
        # seg_map[edges == 1] = 2
        seg_map = seg_map.data.numpy()
        edges = cv2.Canny(seg_map.astype(np.uint8), 0.1,0.2)
        kernel = np.ones((5,5), dtype = np.uint8)
        erosion = cv2.erode(edges,kernel,iterations = 1)
        dilation = cv2.dilate(edges,kernel,iterations = 1)
        seg_map[dilation != 0] = 2
        
        #resize
        if not resize:
            #return numpy array
            seg_map = cv2.resize(seg_map.data.numpy(),(self.width_org, self.height_org) )
        
        if img_show:
            plt.figure(figsize=(10,10))
            ax = plt.subplot(1, 1, 1)
            plt.title('Predicted Rooms and walls', fontsize=20)
            ax.axis('off')
            n_rooms = 12
            rseg = ax.imshow(seg_map, cmap = 'rooms', vmin=0, vmax=n_rooms-0.1)
            cbar = plt.colorbar(rseg, ticks=np.arange(n_rooms)+0.3, fraction=0.046, pad=0.01)
            cbar.ax.set_yticklabels(room_classes, fontsize=20)
            plt.show()
            
        

        return seg_map

    
    
    #room detection 
    def room_detect(self, class_id, img_show = True):
        
        self.results = {}

        for i in range(1,11):
            self.results[i] = defaultdict(list)

        for i,lab in enumerate(self.pred['labels'][self.inds]):
            lab = int(lab.data.numpy())
            self.results[lab]['boxes'].append(self.pred['boxes'][i])
        
        im = self.img.data.numpy().copy()
        im = np.moveaxis(im, 0,-1)
        image = im
        
        if len(self.results[class_id]['boxes'])!= 0:
        
            for (x,y,z,w) in self.results[class_id]['boxes']:

                image = cv2.rectangle(image, (x,y), (z,w), (0,252,0), 1) 
                #image = cv2.putText(image, rooms[class_id-1], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)

            result = cv2.UMat.get(image)
            result = cv2.resize(result, (self.width_org, self.height_org))

            if img_show:
                plt.figure(figsize = (20,12))
                plt.imshow(result)
                plt.title(rooms[class_id-1])
        
        else:
            print('No such object')
            result = None
        
        return result
    
    
    def collect_result(self, thres = 0.1, img_show = True):
    
        
        seg_map =self.get_segmap(thres = thres, img_show =False) #.data.numpy()
        
        result = cv2.imread(self.org_img_path)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        for i in range(1,11):
            
            pix = i+1 if i>=2 else i
            m = copy.copy(seg_map)
            m[seg_map!= pix] = 0 
            m = m.astype(np.uint8)

            contours, _ = cv2.findContours(m,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            for tcnt in contours: 

                x,y,w,h = cv2.boundingRect(tcnt)
                result = cv2.rectangle(result, (x,y), (x+w,y+h), (0,252,0), 3) 
            result = cv2.putText(result, rooms[i-1], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (252,0,0), 2)

        
        #result = cv2.UMat.get(result)
       
        if img_show:
            plt.figure(figsize = (20,12))
            plt.imshow(result)
            plt.title('Predict Result collection')

        return result
    
    
    def getBordered(self, image, width):
        bg = np.zeros(image.shape)
        img = image.copy().astype(np.uint8)
        contours, _ = cv2.findContours(img, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        biggest = 0
        bigcontour = None
        for contour in contours:
            area = cv2.contourArea(contour) 
            if area > biggest:
                biggest = area
                bigcontour = contour
        return cv2.drawContours(bg, [bigcontour], 0, (255, 255, 255), width).astype(bool)

    



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
    transforms.append(T.Resize((256,256)))
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

    

    return model, dataset, data_loader


def main():

    #supress warning
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(
        description='MaskRCNN for Cubicasa Dataset')
    

    parser.add_argument('--epoch', type=str, default=None,
                        help="which epoch result model to run, default: None")
    parser.add_argument('--data_name', type=str, default='val',
                        help="which dataset to test, default: val")
    parser.add_argument('--run_all', type=bool, default=False,
                        help="whether to run all models, default: None")
    parser.add_argument('--data_name2', type=str, default=None,
                        help="the name of second dataset, default: None")
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model, dataset, data_loader = prepare_model(args.data_name, device)

    if args.data_name2:
        model, dataset_test, data_loader_test = prepare_model(args.data_name2, device)

    evaluator = runningScore(12)

    if args.epoch:
        model.load_state_dict(torch.load(f'models/maskrcnn_{args.epoch}.pt',map_location='cpu'))
        evaluate(model, data_loader, device=device)

        for idx in tqdm(range(len(dataset))):
            
            dm = Decode_Maskrcnn(dataset, idx, model, nms = 1)
    
            seg_gt = torch.as_tensor(dm.get_groundtruth(resize = True, img_show = False))
            seg_pred = torch.as_tensor(dm.get_segmap(0.1, resize = True, img_show = False))
            evaluator.update(seg_gt,seg_pred)
        evaluator.get_scores()
    
    if args.run_all:
        for epoch in range(5):
            model.load_state_dict(torch.load(f'checkpoints/maskrcnn_{epoch}_resized.pt',map_location='cuda' if torch.cuda.is_available() else 'cpu'))
            
            for idx in range(len(dataset)):
                dm = Decode_Maskrcnn(dataset, idx, model, nms = 1)
                seg_gt = torch.as_tensor(dm.get_groundtruth(resize = True, img_show = False))
                seg_pred = torch.as_tensor(dm.get_segmap(0.1, resize = True, img_show = False))
                evaluator.update(seg_gt,seg_pred)
            print('*'*25+f'validation result of epoch {epoch}'+'*'*25)
            print(evaluator.get_scores())


            if args.data_name2:

                for idx in range(len(dataset_test)):
        
                    dm = Decode_Maskrcnn(dataset_test, idx, model, nms = 1)
                    seg_gt = torch.as_tensor(dm.get_groundtruth(resize = True, img_show = False))
                    seg_pred = torch.as_tensor(dm.get_segmap(0.1, resize = True, img_show = False))
                    evaluator.update(seg_gt,seg_pred)
                print('*'*25+f'test result of epoch {epoch}'+'*'*25)
                print(evaluator.get_scores())
        
        # for epoch in range(1,5):
        #     model.load_state_dict(torch.load(f'models/maskrcnn_{epoch}_resized.pt',map_location='cpu'))
        #     evaluate(model, data_loader, device=device)
        #     print('*'*25+f'validation result of epoch {epoch}'+'*'*25+'finished')


        #     if args.data_name2:
        #         evaluate(model, data_loader_test, device=device)
        #         print('*'*25+f'test result of epoch {epoch}'+'*'*25+'finished')





main()


    

