
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
import torchvision 
import torchvision.transforms as transforms
from torch import optim
import os 
import numpy as np
import cv2
import glob
from PIL import Image
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = [    
    "JUMPING",
    "JUMPING_JACKS",
    "BOXING",
    "WAVING_2HANDS",
    "WAVING_1HAND",
    "CLAPPING_HANDS"

] 


number = [*range(0, len(LABELS), 1)]
classes=zip(number,LABELS)
classes=dict(classes)



batch_size=28
input_length=36
sequence_length=32
number_layers=4
num_classes=len(classes)
hidden_size=256
epoches=10

print(num_classes)



class Videodataset(Dataset):
    """video Landmarks dataset."""

    def __init__(self, root_dir,sequence_length, transform=None ,training =True):
        self.sequence_length=sequence_length
        self.root_dir = root_dir
        
        X_train_path = self.root_dir + "X_train.txt"
        X_test_path = self.root_dir + "X_test.txt"

        y_train_path = self.root_dir + "Y_train.txt"
        y_test_path = self.root_dir + "Y_test.txt"
        self.y_data=None
        self.x_data=None
        if training ==True:
          with open(X_train_path, 'r') as file_:
            self.x_data = np.array(
                [elem for elem in [
                    row.split(',') for row in file_
                ]], 
                dtype=np.float32
            )
            self.x_data[:,0::2]=(self.x_data[:, 0::2]-320)/640
            self.x_data[:,1::2]=(self.x_data[:, 1::2]-240)/480
          blocks = int(len(self.x_data) / self.sequence_length)
    
          self.x_data = np.array(np.split(self.x_data,blocks))
          
          with open(y_train_path, 'r') as file_:
            self.y_data = np.array(
                [elem for elem in [
                    row.replace('  ', ' ').strip().split(' ') for row in file_
                ]], dtype=np.float32)
            self.y_data=self.y_data-1
        else:
          with open(X_test_path, 'r') as file_:
            print("Testing data")
            self.x_data = np.array(
                [elem for elem in [
                    row.split(',') for row in file_
                ]], 
                dtype=np.float32
            )
            self.x_data[:,0::2]=(self.x_data[:, 0::2]-320)/640
            self.x_data[:,1::2]=(self.x_data[:, 1::2]-240)/480
          blocks = int(len(self.x_data) / self.sequence_length)
    
          self.x_data = np.array(np.split(self.x_data,blocks))

          with open(y_test_path, 'r') as file_:
            self.y_data = np.array(
                [elem for elem in [
                    row.replace('  ', ' ').strip().split(' ') for row in file_
                ]], 
                dtype=np.int32
            )
            self.y_data=self.y_data-1
    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        key_point_data,label= self.x_data[idx],self.y_data[idx]
        # #print(video_samples.shape)
        sample = {'keypoint': torch.tensor(key_point_data),"label":label}

        return sample
    
    
    
train_Dataset=Videodataset(root_dir="./data/RNN-HAR-2D-Pose-database/",sequence_length=sequence_length,training=True)
test_Dataset=Videodataset(root_dir="./data/RNN-HAR-2D-Pose-database/",sequence_length=sequence_length,training=False)

    
train_Dataloader=DataLoader(train_Dataset,batch_size=batch_size,shuffle=True)
test_Dataloader=DataLoader(test_Dataset,batch_size=batch_size,shuffle=False)

    
    
    
    
    
    

class LSTM_model(nn.Module):
  def __init__(self, input_,hidden_size,num_layers,output_number) -> None:
    super(LSTM_model, self).__init__()
    self.input_=input_
    self.hidden_size=hidden_size
    self.num_layers=num_layers
    self.output_number=output_number
    self.lstm_layer=nn.LSTM(self.input_,self.hidden_size,self.num_layers,batch_first=True)
    self.fc=nn.Linear(self.hidden_size*sequence_length,self.output_number)

  def forward(self,x,h0,c0):
    out,(hn,cn)=self.lstm_layer(x,(h0,c0))
    # #print("out shape given before",out.shape)
    out=out.reshape(out.shape[0],-1)
    # #print("out shape given after",out.shape)
    return self.fc(out)

  def init(self,batch_size):
    h0=torch.zeros(self.num_layers,batch_size,self.hidden_size).to(device)
    c0=torch.zeros(self.num_layers,batch_size,self.hidden_size).to(device)
    return h0,c0

model=LSTM_model(input_=input_length,hidden_size=256,num_layers=4,output_number=num_classes).to(device)

lr=0.001
loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)



def train_one_epoch(epoch_index):
    running_loss = 0.
    correct = 0
    loss_value=0
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for value, data in enumerate(train_Dataloader):
        # Every data instance is an input + label pair
        inputs, labels = (data["keypoint"]).to(device),(data["label"]).to(device)

        # # Zero your gradients for every batch!
        optimizer.zero_grad()
        h0,c0=model.init(batch_size=inputs.shape[0])
        # # Make predictions for this batch
        outputs = model(inputs,h0,c0)

        # #print("predicted shape",outputs.shape)
        # #print("actual shape",labels.shape)
        # # Compute the loss and its gradients
        labels=torch.squeeze(labels,1)
        labels = labels.type(torch.LongTensor)
        labels=labels.to(device)
        loss = loss_fn(outputs, labels)
        loss.backward()

        # # Adjust learning weights
        optimizer.step()

        # Gather data and report

          
        outputs = outputs.cpu().data.numpy()
        labels = labels.cpu().data.numpy()
        
#         correct += (new_mat == labels).sum()
        arg_max_outputs=outputs.argmax(axis=1)
#         #print(outputs)
#         #print(outputs.argmax(axis=1))
#         #print(labels)
#         #print(labels.argmax(axis=1))
        correct+=(arg_max_outputs==labels).sum()

        running_loss += loss.item()
        # print("value of j",value)
        if (value+1)%100==0:
            loss_value = running_loss / 100 # loss per batch
            # tb_x = epoch_index * len(train_Dataloader) + i + 1
            
            accuracy = 100 * correct / (100*(len(labels)))

            print("Accuracy = {} and loss = {} at batch {} and epoch {}".format(accuracy,loss_value,value,epoch_index))
            running_loss = 0.
            correct = 0
            loss_value=0
            running_loss = 0.
    torch.save({'epoch': epoch_index,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': loss}, "./checkpoints/checkpoints_v5.pt")
    return loss_value



# model.load_state_dict(torch.load("./model_trained/lstmmodel_v5.pth"))
# model.eval()


def test_data():
    running_loss = 0.
    correct = 0
    loss_value=0
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for value, data in enumerate(test_Dataloader):
        # Every data instance is an input + label pair
        inputs, labels = (data["keypoint"]).to(device),(data["label"]).to(device)
        # # Zero your gradients for every batch!
        h0,c0=model.init(batch_size=inputs.shape[0])
        # # Make predictions for this batch
        outputs = model(inputs,h0,c0)
        # #print("predicted shape",outputs.shape)
        # #print("actual shape",labels.shape)
        # # Compute the loss and its gradients

        labels=torch.squeeze(labels,1)
        labels = labels.type(torch.LongTensor).to(device)
        loss = loss_fn(outputs, labels)
        
        outputs = outputs.cpu().data.numpy()
        labels = labels.cpu().data.numpy()
        

        arg_max_outputs=outputs.argmax(axis=1)
        correct+=(arg_max_outputs==labels).sum()

        running_loss += loss.item()
        # print("value of j",value)
        if (value+1)%50==0:
            loss_value = running_loss / 50
            accuracy = 100 * correct / (50*(len(labels)))

            print("Accuracy = {} and loss = {} at batch {}".format(accuracy,loss_value,value))
            running_loss = 0.
            correct = 0
            loss_value=0
            running_loss = 0.
            
    return loss_value

for epoch in range(epoches):
    train_one_epoch(epoch)

torch.save(model.state_dict(), "./model_trained/lstmmodel_v5.pth")

test_data()
