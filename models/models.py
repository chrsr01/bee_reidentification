import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from triplet_loss import TripletLoss

class SimpleCNNv2(nn.Module):
    def __init__(self, input_shape=(3, 56, 56), conv_blocks=2, latent_dim=128, l2_norm=True, dropout=True):
        super(SimpleCNNv2, self).__init__()
        self.conv_blocks = conv_blocks
        self.l2_norm = l2_norm
        self.dropout = dropout

        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv_block_layers = nn.ModuleList()
        for _ in range(conv_blocks - 1):
            block = nn.ModuleDict({
                'pool': nn.MaxPool2d(2, 2),
                'bn1': nn.BatchNorm2d(64),
                'conv1': nn.Conv2d(64, 64, kernel_size=3, padding=1),
                'bn2': nn.BatchNorm2d(64),
                'conv2': nn.Conv2d(64, 64, kernel_size=3, padding=1),
            })
        self.conv_block_layers.append(block)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * (input_shape[1] // (2**(conv_blocks - 1))) * (input_shape[2] // (2**(conv_blocks - 1))), latent_dim)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        for i in range(self.conv_blocks - 1):
            block = self.conv_block_layers[i]
            xp = block['pool'](x)
            x = F.relu(block['bn1'](xp))
            x = F.relu(block['bn2'](block['conv1'](x)))
            x = block['conv2'](x)
            x = xp + x
            x = self.flatten(x)
            if self.dropout:
                x = self.dropout1(x)
                x = self.fc(x)
            if self.dropout:
                x = self.dropout2(x)
            if self.l2_norm:
                x = F.normalize(x, p=2, dim=1)
        return x

class SimpleCNNv2Lightning(pl.LightningModule):
    def __init__(self, input_shape=(3, 56, 56), conv_blocks=3, latent_dim=128, l2_norm=True, dropout=True, lr=0.001,loss_fn = TripletLoss(device="cuda")):
        super(SimpleCNNv2Lightning, self).__init__()
        self.conv_blocks = conv_blocks
        self.l2_norm = l2_norm
        self.dropout = dropout
        self.lr = lr
        self.loss_fn = loss_fn
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv_block_layers = nn.ModuleList()
        for _ in range(conv_blocks - 1):
            block = nn.ModuleDict({
                'pool': nn.MaxPool2d(2, 2),
                'bn1': nn.BatchNorm2d(64),
                'conv1': nn.Conv2d(64, 64, kernel_size=3, padding=1),
                'bn2': nn.BatchNorm2d(64),
                'conv2': nn.Conv2d(64, 64, kernel_size=3, padding=1),
            })
            self.conv_block_layers.append(block)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * (input_shape[1] // (2**(conv_blocks - 1))) * (input_shape[2] // (2**(conv_blocks - 1))), latent_dim)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        for i in range(self.conv_blocks - 1):
            block = self.conv_block_layers[i]
            xp = block['pool'](x)
            x = F.relu(block['bn1'](xp))
            x = F.relu(block['bn2'](block['conv1'](x)))
            x = block['conv2'](x)
            x = xp + x
        
        x = self.flatten(x)
        if self.dropout:
            x = self.dropout1(x)
        x = self.fc(x)
        if self.dropout:
            x = self.dropout2(x)
        if self.l2_norm:
            x = F.normalize(x, p=2, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss',loss)
        print('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y
    

class MeanAggLayer(nn.Module):
    def forward(self,x):
        return torch.mean(x,dim=1)

class TrackDistanceLoss(nn.Module):
    def __init__(self, margin = 0.2):
        super(TrackDistanceLoss, self).__init__()
        self.margin = margin
        self.losses =[]
        
    def forward(self, inputs):
        print(inputs.shape)
        track_distances = -torch.matmul(inputs, inputs.transpose(-2,-1)) + 1.0
        max_dists = torch.max(track_distances,dim=1).values
        max_dists = torch.max(track_distances,dim=2).values
        track_max_dist = torch.max(torch.mean(max_dists), torch.tensor(self.margin, device=inputs.device))
        self.add_loss(track_max_dist)
        return inputs
    
    def add_loss(self, loss):
        if not hasattr(self, 'losses'):
            self.losses = []
        self.losses.append(loss)
        
    def get_loss(self):
        return sum(self.losses)

    def reset_losses(self):
        self.losses = []
        
        
class L2Normalize(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=1)
       
class TrackModel(pl.LightningModule):
    def __init__(self, backbone, name, agg_layer = MeanAggLayer(), track_len = 4, margin = 2.0):
        super(TrackModel,self).__init__()
        self.name = name
        self.track_len = track_len
        self.agg_layer = agg_layer
        self.margin = margin
        self.track_distance_loss = TrackDistanceLoss(margin = margin)
        self.backbone = backbone
        self.l2_normalize = L2Normalize()
    
    def forward(self,x):
        x = torch.stack([self.backbone(x[:,i,:,:,:]) for i in range(self.track_len)], dim= 1)
        x = self.track_distance_loss(x)
        x = self.agg_layer(x)
        x = self.l2_normalize(x)
        return x
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.track_distance_loss(outputs)
        self.log('train_loss',loss)
        self.track_distance_loss.reset_losses()
        return loss
    
    def validation_step(self,batch,batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.track_distance_loss(outputs)
        self.log('val_loss',loss)
        self.track_distance_loss.reset_losses()
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=1e-03)
        return optimizer
    
    