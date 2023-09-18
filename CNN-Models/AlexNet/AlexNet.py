from uu import Error
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torchvision
import torchvision.transforms as transforms
import os
import random
import torch.nn as nn
from PIL import Image
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
from datetime import datetime
#Image Dataset Creator
class ImageDataset(Dataset):

    def __init__(self, image_dict, transform=None):
        self.image_dict = image_dict
        self.transform = transform

    def __len__(self):
        return len(self.image_dict['images'])

    def __getitem__(self, idx):
        image = self.image_dict['images'][idx]
        label = self.image_dict['labels'][idx]

        if self.transform:
            image = self.transform(image)

        # Convert label to tensor
        label = torch.tensor(int(label))

        return image, label

#Image preprocessor
class ImageTransformer:

    def __init__(self):
        self._img_size = 256
        self._img_crop_size = 224
        self._mean = [0.485, 0.456, 0.406]
        self._std = [0.229, 0.224, 0.225]
        self._required_channels = "RGB"

        self.transform = transforms.Compose([
            transforms.Resize(self._img_size),
            transforms.CenterCrop(self._img_crop_size),
            transforms.Lambda(lambda x: x.convert(self._required_channels) if x.mode != self._required_channels else x),
            transforms.ToTensor(),
            transforms.Normalize(mean=self._mean, std=self._std)
        ])


class TrainModel():

    def __init__(self, parameters):
        self.num_classes = parameters["num_classes"]
        self.dropout= parameters["dropout"]
        self.Epochs = parameters["numEpochs"]
        self.batchSize = parameters["batchSize"]
        self.learningRate = parameters["learningRate"]
        self.optimizer_Fn = parameters["optimizer"]
        self.num_steps = parameters["num_steps"]
        self.hyperparameterTuning = parameters["hyperparameterTuning"]
        self.LearningRateSchedulerFlag = parameters["LearningRateSchedulerFlag"]
        self.EarlyStopping = parameters["EarlyStopping"]
        self.ErrorAnalysis = parameters["ErrorAnalysis"]
        self.saveModelFormat = parameters["saveModelFormat"]
        self.SaveModelIterFreq = parameters["SaveModelIterFreq"]
        self.metricsSavingIterFreq = parameters["metricsSavingIterFreq"]
        self.evaluationIterFreq = parameters["evaluationIterFreq"]
        self.evaluationMetric = parameters["evaluationMetric"]

    def dataset(self):
        self.dataset_path = "D:\OneDrive - Newgen\IDP_Models\Trade_finance_small_dataset"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def model_definition(self):
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained = True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier[6] = torch.nn.Linear(4096, self.num_classes)
        self.model.to(self.device)


    def classes(self):
        self.classes = os.listdir(self.dataset_path)
        self.label_dict = {value: idx for idx, value in enumerate(self.classes)}
        self.id_label_dict = {idx: value for idx, value in enumerate(self.classes)}


    def train_test_splitter(self):
        self.train_images_dict = {
            'images': [],  # List of train images as PIL images
            'labels': []  # List of corresponding labels (strings) for train images
        }

        self.test_images_dict = {
            'images': [],
            'labels': []
        }  
        for document in os.listdir(self.dataset_path):
            random.seed(10)
            document_path = os.path.join(self.dataset_path,document)
            files= os.listdir(os.path.join(self.dataset_path,document))
            random.shuffle(files)
            for train_file in files[:-int(0.2 * len(files))]:
                img = Image.open(os.path.join(document_path, train_file)).convert("RGB")
                self.train_images_dict["images"].append(img)
                self.train_images_dict["labels"].append(self.label_dict[document])
            for test_file in files[-int(0.2 * len(files)):]:
                img = Image.open(os.path.join(document_path, test_file)).convert("RGB")
                self.test_images_dict["images"].append(img)
                self.test_images_dict["labels"].append(self.label_dict[document])
        
        # Create the data loaders for train, test, and validation
        train_dataset = ImageDataset(self.train_images_dict, transform=ImageTransformer().transform)
        test_dataset = ImageDataset(self.test_images_dict, transform=ImageTransformer().transform)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batchSize, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batchSize, shuffle=False)

    def training_parameters(self):
        #we need to find an approach to set loss function and optimizer
        self.loss_fn = nn.CrossEntropyLoss()
        if self.optimizer_Fn == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate) 
        if self.optimizer_Fn == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learningRate) 
        if self.optimizer_Fn == "RMSprop":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate)     
        if self.LearningRateSchedulerFlag:
            self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)         

    def train_one_epoch(self, epoch_index):
        running_loss = 0.
        last_loss = 0.
        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(self.train_loader):    
        #print(data)
        # Every data instance is an input + label pair
            inputs, labels = data
            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            output = self.model(inputs)
            # Compute the loss and its gradients
            loss = self.loss_fn(output, labels)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
        last_loss = running_loss/i+1 # loss per batch

        return last_loss    
    def calculate_test_loss(self):
        epoch_number = 0
        best_vloss = 1000000.
        epochs_no_improve = 0
        max_epochs_stop = 5
        for epoch in range(self.Epochs):
            #print('EPOCH {}:'.format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self.train_one_epoch(epoch_number)
            if self.LearningRateSchedulerFlag:
                self.scheduler.step()

            # We don't need gradients on to do reporting
            self.model.train(False)

            running_test_loss = 0.0
            for i, test_data in enumerate(self.test_loader):
                test_inputs, test_labels = test_data
                test_outputs = self.model(test_inputs)
                test_loss = self.loss_fn(test_outputs, test_labels)
                running_test_loss += test_loss

            avg_test_loss = running_test_loss / (i + 1)
            if self.EarlyStopping:
                if avg_test_loss < best_vloss:
                    best_vloss = avg_test_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                
                if epochs_no_improve == max_epochs_stop:
                    print("Early stopping triggered - no improvement in validation metric.")
                    break
            print('LOSS train {} test {}'.format(avg_loss, avg_test_loss))
            epoch_number += 1
            if epoch_number % self.SaveModelIterFreq == 0:
                path = "AlexnetModels/model_{}epochs".format(epoch_number) + self.saveModelFormat
                torch.save(self.model, path) #need to work out how to save format
    
    def train_model(self):
        self.dataset()
        self.model_definition()
        self.classes()
        self.train_test_splitter()
        self.training_parameters()
        self.calculate_test_loss()

        
#main

parameters = {"num_classes": 5, 
              "dropout": 0,
              "numEpochs": 10,
              "batchSize": 5,
              "learningRate": 0.01,
              "optimizer": "Adam",
              "num_steps": 10,
              "hyperparameterTuning": True,
              "LearningRateSchedulerFlag": True,
              "EarlyStopping": False,
              "ErrorAnalysis": True,
              "saveModelFormat": ".pt",
              "SaveModelIterFreq": 5,
              "metricsSavingIterFreq": 10,
              "evaluationIterFreq": 10,
              "evaluationMetric": "F1"
              }

t1 = TrainModel(parameters)           
t1.train_model()




          

