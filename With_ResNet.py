import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, random_split

import os
import pathlib
from PIL import Image
import random
import matplotlib.pyplot as plt
import numpy as np

import time

# Siamese Network
class SiameseNetwork(nn.Module):
    def __init__(self, freeze_parameter, initial_weight):
        super(SiameseNetwork, self).__init__()
        # get resnet model
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # fix pretrained parameters or not
        # True = not freeze
        # False = freeze
        for param in self.model.parameters():
            param.requires_grad = freeze_parameter

        self.fc_in_features = self.model.fc.in_features

        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

        # initialize the weights
        if initial_weight:
            self.model.apply(self.init_weights)
            self.fc.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.model(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

# contrastive Loss Implementation
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss

# get image data
class SiameseDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.paths = list(pathlib.Path(path).glob("*/*.jpg"))
        self.transform = transform
        self.classes, self.class_to_idx = self.find_class(path)
    
    def find_class(self, class_path):
            class_names = os.listdir(class_path)
            class_to_idx = {name: i for i, name in enumerate(class_names)}
            return class_names, class_to_idx
    
    def __getitem__(self, index):
        # randomly pick a class
        selected_class = self.classes[random.randint(0, len(self.classes))-1]

        # img_path = ./Dataset_path/selected_class_folder
        selected_class_path = self.path + '/' + str(selected_class)

        # list of the image file names in the class
        images_selected_class = os.listdir(selected_class_path)

        # randomly pick a image index
        random_image_1_index = random.randint(0, len(images_selected_class)-1)
        random_image_1_name = images_selected_class[random_image_1_index]

        # get the first image
        image_1 = Image.open(os.path.join(selected_class_path, random_image_1_name))

        # same class
        if index % 2 == 0:
            # randomly pick a image index
            random_image_2_index = random.randint(0, len(images_selected_class)-1)

            # ensure that the index of the second image isn't the same as the first image
            while random_image_2_index == random_image_1_index:
                random_image_2_index = random.randint(0, len(images_selected_class)-1)
            
            # get the second image
            random_image_2_name = images_selected_class[random_image_2_index]
            image_2 = Image.open(os.path.join(selected_class_path, random_image_2_name))

            # set the label for this example to be positive (1)
            target = torch.tensor(1, dtype=torch.float)
        
        # different class
        else:
            # randomly pick a class
            other_selected_class = self.classes[random.randint(0, len(self.classes))-1]

            # ensure that the class of the second image isn't the same as the first image
            while other_selected_class == selected_class:
                other_selected_class = self.classes[random.randint(0, len(self.classes))-1]
            
            # img_path = ./Dataset_path/selected_class_folder
            other_selected_class_path = self.path + '/' + str(other_selected_class)

            # list of the image file names in the class
            images_other_selected_class = os.listdir(other_selected_class_path)

            # randomly pick a image index
            random_image_2_index = random.randint(0, len(images_other_selected_class)-1)
            random_image_2_name = images_other_selected_class[random_image_2_index]

            # get the first image
            image_2 = Image.open(os.path.join(other_selected_class_path, random_image_2_name))

            # set the label for this example to be positive (0)
            target = torch.tensor(0, dtype=torch.float)

        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        return image_1, image_2, target

    
    def __len__(self):
        return len(self.paths)

# data transform values
# return composed transform
def transform (image_size):
    # Resnet18 Normalization values
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    composed = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                ])
    return composed

# train model
# return training loss list
def train (model, train_loader, optimizer):
    loss_sublist = []

    for batch_index, (image_1, image_2, targets) in enumerate(train_loader):
        image_1, image_2, targets = image_1.to(device), image_2.to(device), targets.to(device)
        model.train()
        optimizer.zero_grad()
        output1, output2 = model(image_1, image_2)
        loss = criterion(output1, output2, targets)

        loss_sublist.append(loss.item())

        loss.backward()
        optimizer.step()
    return loss_sublist

# validate model
# return validation accuracy list
def validate (model, validation_loader, num_validation_sample):
    correct = 0
    with torch.no_grad():
        for (image_1, image_2, targets) in validation_loader:
            image_1, image_2, targets = image_1.to(device), image_2.to(device), targets.to(device)
            model.eval()
            output1, output2 = model(image_1, image_2)

            # eucledian_distance is 1D tensor
            eucledian_distance = nn.functional.pairwise_distance(output1, output2)

            # prediction outputs 1 when distance < 0.5, outputs 0 when >= 0.5
            predict = torch.where(eucledian_distance < 0.5, 1, 0)

            #          output True as 1 or False as 0
            correct += predict.eq(targets.view_as(predict)).item()
    
    accuracy = correct / num_validation_sample
    return accuracy

# create data loader from dataset
# return (train_loader, validation_loader)
def create_data_loader(dataset_path, batch):
    orig_dataset = SiameseDataset(dataset_path, transform=transform(resize))

    size_train = int(0.9 * len(orig_dataset))
    size_validation = len(orig_dataset) - size_train
    train_dataset, validation_dataset = random_split(orig_dataset, [size_train, size_validation])

    # Load datasets into DataLoader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch, shuffle=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=1)
    return train_loader, validation_loader

# function for saving trained model
def save_model(model, save_path, save_model_name):
        save_model_name = save_model_name + '.pt'
        torch.save(model.state_dict(), os.path.join(save_path, save_model_name))

# plot training loss and validation accuracy
def plot_stuff(loss, accuracy, save_path, save_plot, title, file_name):    
    fig, ax1 = plt.subplots(figsize=(10,6))
    color = 'tab:red'
    ax1.plot(loss, color = color)
    ax1.set_xlabel('Iteration', color = color)
    ax1.set_ylabel('total loss', color = color)
    ax1.tick_params(axis = 'y', color = color)
    
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('accuracy', color = color)  # already handled the x-label with ax1
    ax2.plot(accuracy, color = color)
    ax2.tick_params(axis = 'y', color = color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    plt.subplots_adjust(top = 0.6)
    plt.title(title)
    if save_plot:
        plt.savefig(os.path.join(save_path, file_name + '.jpg'))
    plt.show()

#================================================================
#---------------------------settings-----------------------------
# fix pretrained parameters or no
# True = not freeze, False = freeze
freeze_parameter = True
# initialize the weights
initialize_weight = False
# learning rate scheduler
lr_scheduler = True
base_learning_rate = 0.001
max_learning_rate = 0.01
learning_rate_step = 10
lr_mode = "triangular2"
# image resize when transform
resize = 224
# batch size for training
batch = 5
# epochs
epochs = 25
# set device
device = torch.device("cuda")
# save loss and accuracy plot
save_plot = False
# save trained model
save_trained_model = False
# dataset path
dataset_path = './Dataset'
# save file path
save_path = './Training_Result/'
#---------------------------settings-----------------------------
#================================================================


#=======================================================================================
#----------------------------------------main-------------------------------------------
start_time = time.time()

# data loaders
train_loader = create_data_loader(dataset_path, batch)[0]
validation_loader = create_data_loader(dataset_path, batch)[1]

# loss criterion
criterion = ContrastiveLoss()

# model
model = SiameseNetwork(freeze_parameter, initialize_weight).to(device)

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=base_learning_rate)

# Dynamic learning rate
if lr_scheduler:
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                  base_lr=base_learning_rate,
                                                  max_lr=max_learning_rate,
                                                  step_size_up=learning_rate_step,
                                                  mode=lr_mode)

# data for performance analysis
loss_list = []
accuracy_list = []
num_validation_sample = len(validation_loader.dataset)

# Train and validate model
for epoch in range(epochs):
    # train 
    loss_sublist = train(model, train_loader, optimizer)
    # update trainning loss for each epoch
    loss_list.append(np.mean(loss_sublist))

    # validate
    accuracy = validate(model, validation_loader, num_validation_sample)
    # update validation accuracy for each epoch
    accuracy_list.append(accuracy)

    if lr_scheduler:
        scheduler.step()

    print('loop ', epoch + 1, ' done')

end_time = time.time()
run_time_spent = end_time - start_time
print("The time of execution of the program is :", run_time_spent, "seconds")

settings_name_for_file = ("Resnet18" + 
            "_FreezeParameter" + str(freeze_parameter) +
            "_InitializeWeight" + str(initialize_weight) +
            "_Resize" + str(resize) +
            "_Epochs" + str(epochs) +
            "_Batch" + str(batch) +
            "_lrScheduler" + str(lr_scheduler) +
            "_LR" + str(base_learning_rate) + "to" + str(max_learning_rate) +
            "_LRstep" + str(learning_rate_step) +
            "_LRmode" + str(lr_mode))

settings_name_for_plot_title = ("Model used: Resnet18" + 
            "\nFreeze Parameter: " + str(freeze_parameter) +
            "\nInitialize Weights: " + str(initialize_weight) +
            "\nResize Image: " + str(resize) +
            "\nEpochs: " + str(epochs) +
            "\nBatch: " + str(batch) +
            "\nLearning Rate Scheduler: " + str(lr_scheduler) +
            "\nLearning Rate: " + str(base_learning_rate) + " to " + str(max_learning_rate) +
            "\nLearning Rate Step: " + str(learning_rate_step) +
            "\nLearning Rate Mode: " + str(lr_mode) + 
            "\nTime Spent: " + str(run_time_spent))

plot_stuff(loss_list, accuracy_list, save_path, save_plot, settings_name_for_plot_title, settings_name_for_file)

if save_trained_model:
    save_model(model, save_path, settings_name_for_file)

#----------------------------------------main-------------------------------------------
#=======================================================================================