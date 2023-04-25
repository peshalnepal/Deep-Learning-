
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
import torchvision 
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch import optim
import os 
import numpy as np
import cv2
import glob
from PIL import Image
import math
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify, abort
import json                    
import base64                  
import logging             
import pickle
import demo_camera

os.remove("./example.log")
logging.basicConfig(filename='example.log', level=logging.DEBUG)


app = Flask(__name__)

number_of_inputs=0
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

taoof=0



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



transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.ToTensor()])



batch_size=1
input_length=36
sequence_length=32
number_layers=4
num_classes=len(classes)
hidden_size=256
width,height= (368,368)
lstm_input=(torch.zeros(size=(batch_size,sequence_length,input_length))).to(device)

def preprocess_image(image):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, (640,480), interpolation = cv2.INTER_AREA)
    return image


def load_data(data_matrix):
    global lstm_input
    lstm_input=torch.roll(lstm_input,-1)
    lstm_input[0][-1]=data_matrix
    return lstm_input
    
def arragne_data(candidate_value, subset_value):
    candidate_arrange = []
    for n in range(len(subset_value)):
        candidate_one = []
        for i in range(18):
            index = int(subset_value[n][i])
            if index == -1:
                candidate_one.extend(np.array([(0.0-320)/640, (0.0-240)/480], dtype=np.float64))
                # continue
            else:
                
                value=np.array([(candidate_value[index][0]-320)/640,(candidate_value[index][1]-240)/480],dtype=np.float64)
                candidate_one.extend(value)
        candidate_arrange.append(np.array(candidate_one, dtype=np.float64))
    tensor_candidate=torch.FloatTensor(np.array(candidate_arrange, dtype=np.float64)).to(device)
    #print((tensor_candidate).shape)
    if len(subset_value)==0:
      tensor_candidate=torch.FloatTensor(np.zeros(shape=(1,36), dtype=np.float64)).to(device)
    return tensor_candidate


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
model.load_state_dict(torch.load("./model_trained/lstmmodel_v5.pth"))
model.eval()

def predict_output(image,h0,c0):
    #print("inputs.shape",inputs.shape)
    candidate,subset=demo_camera.getpose(image)

    lstm_input=load_data(arragne_data(candidate,subset)[0])
    outputs = model(lstm_input,h0,c0)
    return candidate,subset,outputs

@app.route("/", methods=['POST'])
def model_stride():
    video_classify="Can't classify yet"
    try:
        
        if not request.json or 'image' not in request.json: 
            abort(400)
        im_b64 = request.json['image']
        # image_shape=request.json["image_shape"]
        # convert it into bytes  
        

        imdata = base64.b64decode(im_b64)
        im= pickle.loads(imdata)
        image = cv2.imdecode(im, cv2.IMREAD_COLOR)
        # cv2.imwrite("save.jpg",image)
        logging.debug("image saved")
        h0,c0=model.init(batch_size=1)
        # logging.debug(f"original image shape{image.shape}")
        image=preprocess_image(image)
        # logging.debug(f"transformed image shape{image.shape}")
        candidate,subset,outputs =predict_output(image,h0,c0)
        
        arg_max_outputs=outputs.argmax(axis=1)
        video_classify=classes[int(arg_max_outputs)]
        # if outputs[0][int(arg_max_outputs)]>0.1:
        #     video_classify=classes[int(arg_max_outputs)]
        # else:
        #     video_classify="Can't classify yet"
    except Exception as e:
        print(e)
    return jsonify({"stride":video_classify,"candidate":candidate,"subset":subset})
    # return jsonify({"stride":video_classify})

if __name__=="__main__":
    app.run(host='0.0.0.0', port=8000)

