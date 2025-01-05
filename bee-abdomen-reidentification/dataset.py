from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
from PIL import Image
import torch
import h5py
import cv2

def get_abdomen(image):
        img = np.array(image)
        # Check image shape before slicing
        if img.shape[0] == 128:
            return img[68:124, 36:92, :]
        elif img.shape[0] == 256:
            if img.shape[1] >= 240 and img.shape[2] == 3:
                return img[4:60, 4:60, :]
            else:
                raise ValueError("Image shape not suitable for expected slicing")
        else:
            raise ValueError("Unexpected image shape: {}".format(img.shape))
        
class ImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
        
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.annotations.loc[idx]["filename_scaled"])
        image = Image.open(img_name).convert("RGB")
        abdomen = get_abdomen(image)
        label = self.annotations.label.loc[idx]
        image = torch.tensor(abdomen, device = torch.device('cuda'), dtype=torch.float32)
       
        image = image.permute(2,0,1)
        return image, label
    
    
class TensorDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
        
    
    def __getitem__(self, idx):
        tensor_loc = os.path.join(self.img_dir, self.annotations.loc[idx]["tensor_location"])
        tensor = torch.load(tensor_loc)
        label = self.annotations.label.loc[idx]
        return tensor, label

class TrackDataset(Dataset):
    def __init__(self, df, track_len=5, rescale_factor=1, image_augmentation=False, censored=True, label_column="track_tag_id"):
        self.filenames, self.labels = self.extract_filenames_and_labels(df, censored=censored, label_column=label_column)
        self.track_len = track_len
        self.rescale_factor = rescale_factor
        
        # Define the image transformations
      
        
    def extract_filenames_and_labels(self, df, censored, label_column):
        # Implement the logic to extract filenames and labels from the dataframe
        filenames = df['filename'].tolist()
        labels = df[label_column].tolist()
        return filenames, labels
    
    def __len__(self):
        return len(self.filenames) // self.track_len

    def get_abdomen(self,image):
        img = np.array(image)
        if img.shape[0] == 128:
            return img[68:124, 36:92,:]
        else:
            return img[16:240, 16:240,:]

    def __getitem__(self, idx):
        track_start = idx * self.track_len
        track_end = track_start + self.track_len
        track_filenames = self.filenames[track_start:track_end]
        
        images = []
        for filename in track_filenames:
            img = Image.open(f"data/{filename}")
            img =img.resize((128,128))
            img = self.get_abdomen(img)
            img = torch.tensor(img, device=torch.device("cuda"), dtype=torch.float32)
            img = img.permute(2,0,1)
            images.append(img)
        
        images = torch.stack(images)
        label = self.labels[track_start]
        
        return images, label
    
class NonTrackDataset(Dataset):
    def __init__(self, df, track_len=5, rescale_factor=1, image_augmentation=False, censored=True, label_column="track_tag_id"):
        self.filenames, self.labels = self.extract_filenames_and_labels(df, censored=censored, label_column=label_column)
        self.track_len = track_len
        self.rescale_factor = rescale_factor
        
        # Define the image transformations
      
        
    def extract_filenames_and_labels(self, df, censored, label_column):
        # Implement the logic to extract filenames and labels from the dataframe
        filenames = df['filename'].tolist()
        labels = df[label_column].tolist()
        return filenames, labels
    
    def __len__(self):
        return len(self.filenames) // self.track_len

    def get_abdomen(self,image):
        img = np.array(image)
        if img.shape[0] == 128:
            return img[68:124, 36:92,:]
        else:
            return img[16:240, 16:240,:]

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        label = self.labels[idx]
        img = Image.open(f"data/{filename}")
        img = img.resize((128,128))
        img = self.get_abdomen(img)
        img = torch.tensor(img, device=torch.device("cuda"), dtype=torch.float32)
        img = img.permute(2,0,1)
        
        return img, label

class H5Dataset(Dataset):
    def __init__(self, img_file):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_file = img_file

    def __len__(self):
        with h5py.File(self.img_file,"r") as db:
            length = len(db["images"])
        return length
        
    def __getitem__(self, idx):
        with h5py.File(self.img_file,"r") as db:
            tensor = torch.tensor(db["images"][idx,:,:],dtype=torch.float32,device = torch.device('cuda'))
            label = db["label"][idx]
        return tensor, label
    
import cv2
    
# Custom Dataset class
class EvalDataset(Dataset):
    def __init__(self, filenames, rescale_factor=4, image_size=(128, 128), white=False,small_scale = False):
        self.filenames = filenames
        self.rescale_factor = rescale_factor
        self.image_size = image_size
        self.white = white
        self.small_scale = small_scale
      


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join("data", self.filenames[idx])
        image = Image.open(img_path).convert('RGB')
        #image = self.transform(image)
        image =image.resize(self.image_size)
        abdomen = self.get_abdomen(image)#
        if self.small_scale:
            abdomen = cv2.resize(abdomen, (56, 56), interpolation=cv2.INTER_AREA)
        image = torch.tensor(abdomen, device = torch.device('cuda'), dtype=torch.float32)
        image = image.permute(2,0,1)
        return image

    def get_abdomen(self, image):
        img = np.array(image)
        # Check image shape before slicing
        if img.shape[0] == 128:
            abdomen = img[68:124, 36:92, :]
        elif img.shape[0] == 256:
            abdomen = img[132:248, 85:171,:]
        else:
            raise ValueError("Unexpected image shape: {}".format(img.shape))

        # Convert blue parts to white
        # Define the lower and upper bounds for blue color in BGR format
        if self.white:
            lower_blue = np.array([100, 0, 0])
            upper_blue = np.array([255, 50, 50])

            # Convert RGB to BGR
            abdomen_bgr = cv2.cvtColor(abdomen, cv2.COLOR_RGB2BGR)

            # Create a mask for blue color
            mask = cv2.inRange(abdomen_bgr, lower_blue, upper_blue)

            # Change blue to white
            abdomen_bgr[mask != 0] = [255, 255, 255]

            # Convert BGR back to RGB
            abdomen = cv2.cvtColor(abdomen_bgr, cv2.COLOR_BGR2RGB)
        
        return abdomen