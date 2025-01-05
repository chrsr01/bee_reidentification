import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import h5py
from dataset import TrackDataset, NonTrackDataset
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

def prepare_for_triplet_loss_track(df, track_len=4,  repeats=10, label="label"):
    pairs = list()
    pair_labels = list()

    for i in range(repeats):

        ids = df[label].unique()
        shuffle(ids)

        A_df = df.groupby(label).sample(track_len, replace=True)
        A_df = A_df.set_index(label).loc[ids].reset_index()

        B_df = df.groupby(label).sample(track_len, replace=True)
        B_df = B_df.set_index(label).loc[ids].reset_index()

        A = A_df.filename.values
        B = B_df.filename.values
        A_label = A_df[label].values
        B_label = B_df[label].values

        pdf = np.hstack((A.reshape(-1, 1, track_len), B.reshape(-1, 1, track_len)))
        labels = np.dstack((A_label, B_label))
        
        pairs.append(pdf)
        pair_labels.append(labels)

    pair_df = np.vstack(pairs)
    print(pair_df)
    pair_labels = np.vstack(pair_labels)
    df = pd.DataFrame({"filename": pair_df.ravel(), "label": pair_labels.ravel()})
    return df

def train_valid_split_df(df, train_frac=0.8):
    labels = df.label.unique()
    train_num = int(len(labels)*train_frac)
    rand_labels = np.random.permutation(len(labels))
    train_labels = rand_labels[:train_num]
    train_df = df[df.label.isin(train_labels)]
    valid_df = df[~df.label.isin(train_labels)]
    return train_df, valid_df


def load_torch_dataset(df, track_len=5, rescale_factor=1, image_augmentation=False, censored=True, label_column="track_tag_id", batch_size=256, shuffle=True, num_workers=6):
    dataset = NonTrackDataset(df,  rescale_factor=rescale_factor, image_augmentation=image_augmentation, censored=censored, label_column=label_column)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


def prepare_for_triplet_loss(df, label="label"):
    sdf = df.sort_values(label)

    labels = sdf[label].values
    filename = sdf.filename.values
    
    if labels.shape[0] % 2:
        labels = labels[1:]
        filename = filename[1:]
        

    pair_labels = labels.reshape((-1, 2))
    pair_filename = filename.reshape((-1, 2))

    ridx = np.random.permutation(pair_labels.shape[0])

    labels = pair_labels[ridx].ravel()
    filename = pair_filename[ridx].ravel()

    tdf = pd.DataFrame({"filename":filename, "label":labels})
    return tdf



def load_dataset(train_df, valid_df, augmentation=False, label_column="label", shuffle=True):
    
    train_dataset = load_torch_dataset(train_df, rescale_factor=4, label_column=label_column)
    valid_dataset = load_torch_dataset(valid_df, rescale_factor=4, label_column=label_column)
    
    if shuffle:
        train_dataset = train_dataset.shuffle(len(train_df))
        valid_dataset = valid_dataset.shuffle(len(valid_df))
    
    return train_dataset, valid_dataset


def get_dataset(dataset, augmentation=False):
    
    if dataset == "untagged" or dataset == "untagged_augmented":
        label_column = "label"
        dataset_filename = "data/short_term_train.csv"
        untagged_df = pd.read_csv(dataset_filename)
        train_df, valid_df = train_valid_split_df(untagged_df, train_frac=0.8)
        train_df = prepare_for_triplet_loss(train_df)
        valid_df = prepare_for_triplet_loss(valid_df)
        f
    elif dataset == "tagged":
        label_column = "track_tag_id"
        train_csv, valid_csv = "data/long_term_train.csv","data/long_term_valid.csv"
        train_df = pd.read_csv(train_csv)
        train_df = prepare_for_triplet_loss(train_df, label=label_column)
        valid_df = pd.read_csv(valid_csv)
        valid_df = prepare_for_triplet_loss(valid_df, label=label_column)
        
    
    train_dataset, valid_dataset = load_dataset(train_df, valid_df, augmentation=augmentation, label_column="label", shuffle=False)
    return train_dataset, valid_dataset