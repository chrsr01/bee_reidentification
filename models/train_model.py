from models import SimpleCNNv2Lightning
import pytorch_lightning as L
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import H5Dataset
from lightning.pytorch import loggers as pl_loggers
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from evaluation import full_evaluation_df, track_full_evaluation,cmc_evaluation_df

batch_size = 256



def train_model(train_dataset_path, val_dataset_path,finetune = False, model_finetune_path = None):  
    """
    Method to train a model based on a train_dataset path and val_dataset path, The val-dataset is used for monitoring the val_loss of the model during training
    and if necessary early stop the training to prevent overfitting. If a model should be finetuned, provide a model path.
    
    """
    train_dataset = H5Dataset(img_file=train_dataset_path)
    val_dataset = H5Dataset(img_file=val_dataset_path)

    train_loader = DataLoader(train_dataset, shuffle=True,batch_size=batch_size, num_workers=6,persistent_workers=True)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=6,persistent_workers=True)

    if finetune == True:
        model = SimpleCNNv2Lightning.load_from_checkpoint(model_finetune_path)
    else:
        model = SimpleCNNv2Lightning(input_shape=(3,56,56),conv_blocks=3,latent_dim=128)

    early_stopping = EarlyStopping('val_loss', patience=100)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min")

    trainer = L.Trainer(callbacks=[early_stopping,checkpoint_callback],max_epochs=1000,logger = tb_logger,default_root_dir="models",enable_checkpointing=True)
    trainer.fit(model=model, train_dataloaders=train_loader,val_dataloaders=val_loader)

    print(checkpoint_callback.best_model_path)      # prints path to the best model's checkpoint
    print(checkpoint_callback.best_model_score)     # and prints its score

    best_model = SimpleCNNv2Lightning.load_from_checkpoint(checkpoint_callback.best_model_path)
    return best_model
    

def eval_model(model_path,n_distractors):
    """
    Evaluates a given model path with a given number of distractors.  .
    
    """
    model =  SimpleCNNv2Lightning.load_from_checkpoint(model_path)
    n_distractors=n_distractors
    
    tsh_df = pd.read_csv("data/test_same_hour.csv")
    tsh_df = tsh_df[tsh_df.image_id < n_distractors + 2]
    tsh_ranks = cmc_evaluation_df(model, tsh_df)
        
    tddsh_df = pd.read_csv("data/test_different_day_same_hour.csv")
    tddsh_df = tddsh_df[tddsh_df.image_id < n_distractors + 2]
    tddsh_ranks = cmc_evaluation_df(model, tddsh_df)

    tdd_df = pd.read_csv("data/test_different_day.csv")
    tdd_df = tdd_df[tdd_df.image_id < n_distractors + 2]
    tdd_ranks = cmc_evaluation_df(model, tdd_df)
        
    result_dict = {
        "test_same_hour": tsh_ranks,
        "test_different_day_same_hour": tddsh_ranks,
        "test_different_day": tdd_ranks
    }

    result_df = pd.DataFrame(result_dict)
    result_df.to_csv(f"results/results_finetuned_{model_path}.csv")
    


if __name__ == "__main__":
    print("Starting training: ")
    model = train_model("train_data/short_term_train.h5","train_data/short_term_val.h5")
    print("Pre-Training finished")
    finetuned_model = train_model("train_data/long_term_train.h5","train_data/short_term_val.h5")
    print("Finetuning finished")
    eval_model(finetuned_model,10)
