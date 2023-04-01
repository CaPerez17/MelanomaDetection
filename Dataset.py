#Import libraries
import torch # pytorch library for tensors and deep learning
import torchvision #pytoch library for computer vision
import torchvision.transforms as transforms 
from PIL import Image 
import pandas as pd

#Define a custom dataset class that inherits from Dataset
class MelanomaDataset(Dataset) :
    #Initialize the class with a dataframe containing image paths and labels, and a transform function for augmentation
    def __init__(self, dataframe, transform= None) :
        self.dataframe = dataframe # Assign the dataframe to an attribute
        self.transform = transform # Assign the transform function to an attribute

        #Define a method to get the length of the dataset
        def __len__(self):
            return len(self.dataframe) # Return the length of the dataframe

        # Define a method to get an item from the dataset given an index
        def __getitem__(self):
            # Get the image path and label from the dataframe at index idx
            image_path = self.dataframe.iloc[idx]['image_path']
            label = self.dataframe.iloc[idx]['label']

            #Open the image using PIL library 
            image = Image.open(image_path)

            #Apply the transform function if it is not None
            if self.transform:
                image = self.transform(image)

            #Convert the label to a tensor using torch 
            label = torch.tensor(label)

            #Return a dictionary containing the image and label tensors
            return {'image': image, 'label': label}

#Define some augmentation techniques using transforms.compose function
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # Normalize the tensor with mean and standard deviation of ImageNet dataset 
])

# Read the csv file containing image names and labels using pd.read_csv function
df = pd.read_csv('ISIC_2019_Training_GroundTruth.csv')

# Add a new column called image_path that contains the full path of each image using df.apply function
df['image_path'] = df['image'].apply(lambda x: 'ISIC_2019_Training_Input/' + x + '.jpg')

# Drop the columns that are not needed for melanoma classification using df.drop function
df = df.drop(['image', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC'], axis=1)

# Rename the column MEL to label using df.rename function
df = df.rename({'MEL': 'label'}, axis=1)

# Print the first five rows of the dataframe using df.head function
print(df.head())

# Create a dataset object using MelanomaDataset class and passing df and transform as arguments 
dataset = MelanomaDataset(df, transform)

# Create a data loader object using DataLoader class from torch.utils.data library and passing dataset as argument 
dataloader = DataLoader(dataset)