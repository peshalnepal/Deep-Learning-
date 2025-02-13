import torch 
from torch import nn

DarkNet_Architecture=[
    (7,64,2,3),
    "M",
    (3,192,1,1),
    "M",
    (1,128,1,0),
    (3,256,1,1),
    (1,256,1,0),
    (3,512,1,1),
    "M",
    [(1,256,1,0),(3,512,1,1),4],
    (1,512,1,0),
    (3,1024,1,1),
    "M",
    [(1,512,1,0),(3,1024,1,1),2],
    (3,1024,1,1),
    (3,1024,2,1),
    (3,1024,1,1),
    (3,1024,1,1)
]
class Conv_layer(torch.nn.Module):
    def __init__(self,input_channels,output_channels,kernel_size,stride,padding,**kwargs):
        super(Conv_layer,self).__init__()
        self.conv=nn.Conv2d(input_channels,output_channels,kernel_size,stride,padding)
        self.batch_norm=nn.BatchNorm2d(output_channels)
        self.leakyrelu=nn.LeakyReLU(0.1)

    def forward(self,x):
        return self.leakyrelu(self.batch_norm(self.conv(x)))
    
class DarkNet(torch.nn.Module):
    def __init__(self,input_channels,Architecture,**kwargs):
        super(DarkNet,self).__init__()
        self.input_channels=input_channels
        self.Architecture=Architecture
        self.conv_model=self.create_conv_model()
        self.fcs=self.create_flatt_layer(**kwargs)

    def forward(self,x):
        x=self.conv_model(x)
        x=torch.flatten(x,start_dim=1)
        return self.fcs(x)
    
    def create_conv_model(self):
        layers=nn.ModuleList()
        in_channels=self.input_channels
        for idx,architect in enumerate(self.Architecture):
            if type(architect)==tuple:
                layers.append(Conv_layer(in_channels,architect[1],architect[0],architect[2],architect[3]))
                in_channels=architect[1]
            elif architect=="M":
                layers.append(nn.MaxPool2d(2,2))
            else:
                loop=architect[-1]
                layer1=architect[0]
                layer2=architect[1]
                for i in range(loop):
                    layers.append(Conv_layer(in_channels,layer1[1],layer1[0],layer1[2],layer1[3]))
                    layers.append(Conv_layer(layer1[1],layer2[1],layer2[0],layer2[2],layer2[3]))
                    in_channels=layer2[1]
        return nn.Sequential(*layers)
    
    def create_flatt_layer(self,split_size,num_boxes,num_class):
        layers=nn.ModuleList()
        layers.append(nn.Flatten())
        layers.append(nn.Linear(1024*split_size*split_size,4096))
        layers.append(nn.Dropout(0.1))
        layers.append(nn.LeakyReLU(0.1))
        layers.append(nn.Linear(4096,split_size*split_size*(num_class+num_boxes*5)))
        return nn.Sequential(*layers)
    
