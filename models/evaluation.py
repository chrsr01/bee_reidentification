import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from ipywidgets import interact
import pandas as pd
#from beeid2.utils import sensitivity_map
from sklearn.metrics import precision_recall_curve, auc
from skimage import io
import torch
import pytorch_lightning as L
from torch.utils.data import DataLoader
from dataset import EvalDataset



def filename2image(filenames, rescale_factor=4, image_size=(224, 224), batch_size=32, num_workers=6):
    dataset  = EvalDataset(filenames,rescale_factor,image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,prefetch_factor=3)
    return dataloader 


def filename2image_set(filenames, rescale_factor=4, image_size=(128, 128), batch_size=32, num_workers=6,white=False,small_scale = False):
    dataset  = EvalDataset(filenames,rescale_factor=rescale_factor,image_size = image_size,white = white,small_scale=small_scale)
    return dataset

def get_query_gallery(query, query_df, dfGroupedbyTagId, limit=None):
    same_tag = (query_df.track_tag_id == query.track_tag_id)
    different_global_track = (query_df.global_track_id != query.global_track_id)
    same_tag_pool = query_df[same_tag & different_global_track]
    key = same_tag_pool.sample().iloc[0]
    
#     negatives = dfGroupedbyTagId.sample()
    negatives = dfGroupedbyTagId.apply(lambda x: x.iloc[np.random.randint(len(x))])
    different_tag = (negatives.index != query.track_tag_id)
    negatives = negatives[different_tag]
    
    if limit is not None:
        negatives = negatives.sample(limit)
    query_gallery = np.concatenate(([query.filename, key.filename], negatives.filename.values))
    return query_gallery

def compute_distance(query, gallery):
    cos_dist = np.matmul(query, gallery.T)
    euclid_dist = -(cos_dist - 1)
    return euclid_dist

def compute_rank(euclid_dist):
    return np.argmin(np.argsort(euclid_dist))


def split_query_gallery(predictions):
    query = np.expand_dims(predictions[0], axis=0)
    gallery = predictions[1:]
    return query, gallery

def split_queries_galleries(predictions):
    queries_emb = list()
    galleries_emb = list()
    for q_gallery in predictions:
        query, gallery = split_query_gallery(q_gallery)
        queries_emb.append(query)
        galleries_emb.append(gallery)
    return np.array(queries_emb), np.array(galleries_emb)


def cmc_evaluation(model, df, iterations=100, gallery_size=10):
    """
    model: Torch model
    df: a dataframe with the image to evaluate
    
    """
    cdf = df.copy()
    
    query_df = cdf.groupby("track_tag_id").filter(lambda x: len(x["global_track_id"].unique()) > 1)
    dfGroupedbyTagId = cdf.groupby("track_tag_id")
    
    ranks = np.zeros((iterations, gallery_size))

    for it in tqdm(range(iterations)):
        queries = query_df.groupby("track_tag_id").sample()
        queries_and_galleries = list()
        for i, query_data in queries.iterrows():
            query_gallery =  get_query_gallery(query_data, query_df, dfGroupedbyTagId, limit=gallery_size)
            queries_and_galleries.append(query_gallery)

        queries_and_galleries = np.array(queries_and_galleries).ravel()

        images = filename2image(queries_and_galleries)
        predictions = model.predict(images.batch(32))

        query_gallery_size = gallery_size + 2
        queries_emb = predictions[::query_gallery_size]

        pred_idx = np.arange(0, len(predictions))
        galleries_emb = predictions[np.mod(pred_idx, query_gallery_size) != 0]

        queries_emb = queries_emb.reshape(len(queries), 1, -1)
        galleries_emb = galleries_emb.reshape(len(queries), query_gallery_size - 1, -1 )

        # Calucluate distance
        cos_dist = tf.matmul(queries_emb, galleries_emb, transpose_b=True).numpy()
        euclid_dist = -(cos_dist - 1)

        # Calculate Rank
        r = np.argmin(np.argsort(euclid_dist), axis=2)
        r = np.squeeze(r)

        for i in range(gallery_size):
            ranks[it][i] = np.mean(r < (i + 1))
    return np.mean(ranks, axis=0)


def plot_cmc(ranks_means, filename=None):
    x = np.arange(1, len(ranks_means) + 1)
    plt.figure(figsize=(20, 10))
    plt.plot(x, ranks_means, 'o', markersize=8)
    plt.plot(x, ranks_means, 'b-', linewidth=2)
    plt.grid(True)
    plt.ylim(0.0, 1.0)
    plt.yticks(np.arange(0, 1.05, 0.05));
    plt.xticks(np.arange(1, len(ranks_means) + 1, 1));
    plt.xlabel("Rank")
    plt.ylabel("Matching Rate %")
    plt.tick_params(axis='y', which='minor', bottom=False)
    if filename is not None:
        plt.savefig(filename)
        
        
def plot_query_gallery(query_gallery, dist=None, limit=67):
    rows = int(np.ceil(limit/12.0))
    fig, ax = plt.subplots(rows, 12, figsize=(20, 15))
    
    ax = ax.ravel()
    
    for i, image in enumerate(query_gallery):
        ax[i].imshow(image)
        ax[i].set_yticks([])
        ax[i].set_xticks([])
        if i == 0:
            ax[i].set_title("Query")
            ax[i].spines['bottom'].set_color('red')
            ax[i].spines['top'].set_color('red') 
            ax[i].spines['right'].set_color('red')
            ax[i].spines['left'].set_color('red')
            if dist is not None:
                ax[i].set_xlabel("Rank : {}".format(compute_rank(dist)))
        elif i == 1:
            ax[i].set_title("Key")
            ax[i].spines['bottom'].set_color('green')
            ax[i].spines['top'].set_color('green') 
            ax[i].spines['right'].set_color('green')
            ax[i].spines['left'].set_color('green')
            if dist is not None:
                ax[i].set_xlabel("{:.5f}".format(dist[0][i-1]))
        else:
            ax[i].set_title("Distractor")
            if dist is not None:
                ax[i].set_xlabel("{:.9f}".format(dist[0][i-1]))
                

def random_eval_df(df,n_distractors,image_size = (128,128)):
    """
    model: torch model
    df: a dataframe with the image to evaluate
    
    """
    queries_and_galleries = df.filename.values
        
    dataset = filename2image_set(queries_and_galleries, rescale_factor=4, image_size=image_size,num_workers=6)
    batch_size = n_distractors+2
    dataloader= DataLoader(dataset, batch_size=batch_size,prefetch_factor=5,num_workers=6,shuffle=False)
    preds_l=[]
    
    preds = np.array(preds_l)

    query_gallery_size = df.image_id.max() + 1
    n_distractors = query_gallery_size - 2
       
    galleries_per_iteraration = len(df.gallery_id.unique())
    iterations = df.iteration_id.max() + 1
    total_galleries =  galleries_per_iteraration * iterations
    
    queries_emb = preds[::query_gallery_size]
       
    pred_idx = np.arange(0, len(preds))
    galleries_emb = preds[np.mod(pred_idx, query_gallery_size) != 0]

    queries_emb = queries_emb.reshape(total_galleries, 1, -1)
    galleries_emb = galleries_emb.reshape(total_galleries, n_distractors + 1, -1)

    
    euclid_dist = np.random.rand(int(len(dataloader)/batch_size),1,batch_size-1)

    # Calculate Rank
    r = np.argmin(np.argsort(euclid_dist), axis=2)
    r = np.squeeze(r)
    
    ranks = np.zeros(n_distractors)
    for i in range(n_distractors):
        ranks[i] = np.mean(r < (i + 1))
        
    return ranks     
            
def cmc_evaluation_df(model, df, image_size = (128,128), white=False, small_scale = False):
    """
    model: keras model
    df: a dataframe with the image to evaluate
    
    """
    queries_and_galleries = df.filename.values
    
    model.eval()   
    dataset = filename2image_set(queries_and_galleries, rescale_factor=4, image_size=image_size,num_workers=6,white=white,small_scale=small_scale)
        
    batch_size = 256
    dataloader= DataLoader(dataset, batch_size=batch_size,prefetch_factor=5,num_workers=6)
    model.eval()   
    predictions = []

    for batch in tqdm(dataloader):
        with torch.no_grad():
            predictions.append(model(batch))
        
    preds_l=[]
    #print(predictions[0].shape)
    for i in tqdm(range(0, len(predictions))):
        for j in range(0,predictions[i].cpu().numpy().shape[0]):
            preds_l.append(predictions[i][j].cpu().numpy())
    #print(i)
    #preds[i] = predictions[i].cpu()
    preds = np.array(preds_l)

    query_gallery_size = df.image_id.max() + 1
    n_distractors = query_gallery_size - 2
       
    galleries_per_iteraration = len(df.gallery_id.unique())
    iterations = df.iteration_id.max() + 1
    total_galleries =  galleries_per_iteraration * iterations
    
    queries_emb = preds[::query_gallery_size]
       
    pred_idx = np.arange(0, len(preds))
    galleries_emb = preds[np.mod(pred_idx, query_gallery_size) != 0]

    queries_emb = queries_emb.reshape(total_galleries, 1, -1)
    galleries_emb = galleries_emb.reshape(total_galleries, n_distractors + 1, -1 )

    # Calucluate distance
    cos_dist = tf.matmul(queries_emb, galleries_emb, transpose_b=True).numpy()
    euclid_dist = -(cos_dist - 1)
    # Calculate Rank
    r = np.argmin(np.argsort(euclid_dist), axis=2)
    r = np.squeeze(r)
    
    ranks = np.zeros(n_distractors)
    for i in range(n_distractors):
        ranks[i] = np.mean(r < (i + 1))
        
    return ranks

def full_evaluation_df(model, n_distractors=10, plot=False):
    
    # valid_with_shared_ids_df = pd.read_csv(EVALUATION_FILES["valid_with_shared_ids_cmc"])
    # valid_with_shared_ids_df = valid_with_shared_ids_df[valid_with_shared_ids_df.image_id < n_distractors + 2]
    # valid_with_shared_ids_ranks = cmc_evaluation_df(model, valid_with_shared_ids_df)
    # if plot:
    #     plot_cmc(valid_with_shared_ids_ranks)
        
        
    # valid_df = pd.read_csv(EVALUATION_FILES["valid_cmc"])
    # valid_df = valid_df[valid_df.image_id < n_distractors + 2]
    # valid_ranks = cmc_evaluation_df(model, valid_df)
    # if plot:
    #     plot_cmc(valid_ranks)
        
    # test_df = pd.read_csv(EVALUATION_FILES["test_cmc"])
    # test_df = test_df[test_df.image_id < n_distractors + 2]
    # test_ranks = cmc_evaluation_df(model, test_df)
    # if plot:
    #     plot_cmc(test_ranks)
        
    # test_without_overlap_df = pd.read_csv(EVALUATION_FILES["test_no_train_overlap_cmc"])
    # test_without_overlap_df = test_without_overlap_df[test_without_overlap_df.image_id < n_distractors + 2]
    # test_without_overlap_ranks = cmc_evaluation_df(model, test_without_overlap_df)
    # if plot:
    #     plot_cmc(test_without_overlap_ranks)
    
    
    tsh_df = pd.read_csv("data/test_same_hour.csv")
    tsh_df = tsh_df[tsh_df.image_id < n_distractors + 2]
    tsh_ranks = cmc_evaluation_df(model, tsh_df)
    if plot:
        plot_cmc(tsh_ranks)
        
    tddsh_df = pd.read_csv("data/test_different_day_same_hour.csv")
    tddsh_df = tddsh_df[tddsh_df.image_id < n_distractors + 2]
    tddsh_ranks = cmc_evaluation_df(model, tddsh_df)
    if plot:
        plot_cmc(tddsh_ranks)
        
    tdd_df = pd.read_csv("data/test_different_day.csv")
    tdd_df = tdd_df[tdd_df.image_id < n_distractors + 2]
    tdd_ranks = cmc_evaluation_df(model, tdd_df)
    if plot:
        plot_cmc(tdd_ranks)
        
    result_dict = {
        # "valid_with_shared_ids": valid_with_shared_ids_ranks,
        # "valid": valid_ranks,
        # "test": test_ranks,
        # "test": test_without_overlap_ranks,
        "test_same_hour": tsh_ranks,
        "test_different_day_same_hour": tddsh_ranks,
        "test_different_day": tdd_ranks
    }

    result_df = pd.DataFrame(result_dict)
#     result_df.to_csv(output_file)
    return result_df

def full_evaluation(output_file, model, n_distractors=10, plot=False):
    result_df = full_evaluation_df(model, n_distractors=n_distractors, plot=plot)
    result_df.to_csv(output_file, index=False)
    return result_df


def track_full_evaluation(output_file, model, n_distractors=10, plot=False, track_len=4):
    result_df = track_full_evaluation_df(model, n_distractors=n_distractors, plot=plot, track_len=track_len)
    result_df.to_csv(output_file, index=False)
    return result_df


def cmc_track_model_evaluation(track_model, df, track_len=4, batch_size=64, dim=128):
    cddf = df.copy()
    cddf = cddf.sort_values(["iteration_id", "gallery_id", "image_id"])

    test_df = pd.read_csv("data/long_term_valid.csv")
    
    # sample track_len images per track
    tracks = test_df.groupby("global_track_id").sample(track_len, replace=True).sort_values(["global_track_id", "datetime2"])
    
    tracks_ids = tracks["global_track_id"].values[::track_len]
    track_tag_id = tracks["track_tag_id"].values[::track_len]
    datetime = tracks["datetime2"].values[::track_len]
    filenames = tracks["filename"].values
    images = filename2image(filenames)
    trainer = L.Trainer()
    
   # with torch.no_grad():
    #    predictions = trainer.predict(track_model,dataloaders=images)[0].numpy()
    
    
    predictions = track_model(images.batch(track_len).batch(batch_size), verbose=True)
    tracks_emb = pd.DataFrame({"global_track_id": tracks_ids , "emb": list(predictions)})

    query_gallery_size = cddf.image_id.max() + 1
    n_distractors = query_gallery_size - 2

    galleries_per_iteraration = cddf.gallery_id.max() + 1
    iterations = cddf.iteration_id.max() + 1
    total_galleries =  galleries_per_iteraration * iterations

    tracks_emb = cddf.merge(tracks_emb, how="left", on="global_track_id")
    tracks_emb = tracks_emb.sort_values(["iteration_id", "gallery_id", "image_id"])

    queries_emb = tracks_emb[tracks_emb.image_id == 0]
    galleries_emb = tracks_emb[tracks_emb.image_id != 0]

    queries_emb = to_np_array(queries_emb.emb.values, dim=dim)
    galleries_emb = to_np_array(galleries_emb.emb.values, dim)


    queries_emb = queries_emb.reshape(total_galleries, 1, dim)
    galleries_emb = galleries_emb.reshape(total_galleries, n_distractors + 1, dim)

    # Calucluate distance
    cos_dist = tf.matmul(queries_emb, galleries_emb, transpose_b=True).numpy()
    euclid_dist = -(cos_dist - 1)

    # Calculate Rank
    r = np.argmin(np.argsort(euclid_dist), axis=2)
    r = np.squeeze(r)

    ranks = np.zeros(n_distractors)
    for i in range(n_distractors):
        ranks[i] = np.mean(r < (i + 1))

    return ranks
    

def track_full_evaluation_df(model, n_distractors=10, plot=False, track_len=4):
    
    tsh_df = pd.read_csv("data/test_same_hour.csv")
    tsh_df = tsh_df[tsh_df.image_id < n_distractors + 2]
    tsh_ranks = cmc_track_model_evaluation(model, tsh_df, track_len=track_len)
    if plot:
        plot_cmc(tsh_ranks)
        
    tddsh_df = pd.read_csv("data/test_different_day_same_hour.csv")
    tddsh_df = tddsh_df[tddsh_df.image_id < n_distractors + 2]
    tddsh_ranks = cmc_track_model_evaluation(model, tddsh_df, track_len=track_len)
    if plot:
        plot_cmc(tddsh_ranks)
        
    tdd_df = pd.read_csv("data/test_different_day.csv")
    tdd_df = tdd_df[tdd_df.image_id < n_distractors + 2]
    tdd_ranks = cmc_track_model_evaluation(model, tdd_df, track_len=track_len)
    if plot:
        plot_cmc(tdd_ranks)
        
    result_dict = {
        "test_same_hour": tsh_ranks,
        "test_different_day_same_hour": tddsh_ranks,
        "test_different_day": tdd_ranks
    }

    result_df = pd.DataFrame(result_dict)
    result_df.to_csv("results.csv")
    return result_df