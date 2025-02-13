
import torch 
from torch import nn
from torch.nn.modules.conv import Conv2d
import torchvision.transforms as transform
from torchvision.transforms.transforms import Resize
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from PIL import Image as PILImage
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F



class Conv_layer(nn.Module):
  def __init__(self,in_channels,out_channels):
    super(Conv_layer,self).__init__()
    self.model=nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding=1,bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1,bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

  def forward(self,x):
    return self.model(x)

class Unet(nn.Module):
  def __init__(self,in_channels=3,out_channels=1,features=[64,128,256,512]):
    super(Unet,self).__init__()
    self.down_model=nn.ModuleList()
    self.up_model=nn.ModuleList()
    self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
    input_channel=in_channels
    for feature in features:
      self.down_model+=[
                        Conv_layer(input_channel,feature),
                        
      ]
      input_channel=feature
    for feature in reversed(features):
      input_channel=feature
      self.up_model+=[
                      nn.ConvTranspose2d(input_channel*2,feature,kernel_size=2,stride=2),
                      Conv_layer(input_channel*2,feature)
      ]
    
    self.bottle_neck=Conv_layer(features[-1],features[-1]*2)
    self.final_conv=nn.Conv2d(in_channels=features[0],out_channels=out_channels,kernel_size=1)
    self.Sigmoid_=nn.Sigmoid()

  def forward(self,x):
    skip_connections=[]
    for layer in self.down_model:
      x=layer(x)
      skip_connections.append(x)
      x=self.pool(x)
    x=self.bottle_neck(x)
    skip_connections=skip_connections[::-1]
    for idx in range(0,len(self.up_model),2):
      x=self.up_model[idx](x)
      skip_connection=skip_connections[idx//2]
      if x.shape!=skip_connection.shape:
        skip_connection=transform.functional.resize(skip_connection,size=x.shape[2:])
      concate=torch.concat((skip_connection,x),dim=1)
      x=self.up_model[idx+1](concate)

    return self.Sigmoid_(self.final_conv(x))
