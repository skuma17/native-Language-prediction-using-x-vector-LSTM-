import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio

import wavencoder
import torchmetrics
from pytorch_lightning import LightningModule
import pytorch_lightning as pl

from torchmetrics import Accuracy, ConfusionMatrix
from torchmetrics import F1Score, AUROC
from torchmetrics import MeanSquaredError  as MSE
from torchmetrics import MeanAbsoluteError as MAE



from models import Wav2VecLSTM_Base
from models import SpeechBrainLSTM


import pandas as pd
import wavencoder

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))

class LightningModel(pl.LightningModule):
    def __init__(self, HPARAMS):
        super(LightningModel, self).__init__()
        # HPARAMS
        self.save_hyperparameters()
        #self.model = SpeechBrainLSTM(HPARAMS['model_hidden_size'])
        self.model = Wav2VecLSTM_Base(HPARAMS['model_hidden_size'])

        self.ConfutionMatrix_MultiClass_criterion = ConfusionMatrix(num_classes=5)
        self.F1_criterion = F1Score(number_classes=5,
        average="micro")
        self.AUC_criterion = AUROC(num_classes=5,
        average="micro")

        self.nl_numclass = 5

        self.classification_criterion = MSE()
        self.regression_criterion = MSE()
        self.mae_criterion = MAE()
        self.rmse_criterion = RMSELoss()
        self.accuracy = Accuracy()

        self.alpha = HPARAMS['model_alpha']
        self.beta = HPARAMS['model_beta']
        self.gamma = HPARAMS['model_gamma']

        self.lr = HPARAMS['training_lr']

        
        self.csv_path = HPARAMS['speaker_csv_path']
        self.df = pd.read_csv(self.csv_path, sep=' ')
        
        self.a_mean = self.df['Age'].mean()
        self.a_std = self.df['Age'].std()
        
        print(f"Model Details: #Params = {self.count_total_parameters()}\t#Trainable Params = {self.count_trainable_parameters()}")

    def count_total_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
  
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return [optimizer]

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, batch, batch_idx):
        x, y_nl = batch
        y_hat_nl = self(x)

        native_languages_loss = self.cross_entropy_loss(y_hat_nl, y_nl)            
        loss = self.alpha * native_languages_loss

        native_languages_acc = self.accuracy(y_hat_nl.float(), y_nl)
        nl_F1Score = self.F1_criterion(y_hat_nl.float(), y_nl)
        
        
        #nl_confmatrix = self.ConfutionMatrix_MultiClass_criterion.update(y_hat_nl, y_nl )
        #nl_auroc = self.AUC_criterion(y_hat_nl.float(), y_nl)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        return {'loss':loss, 
                'train_native_languages_acc':native_languages_acc,
                'train_nl_F1score':nl_F1Score,                
#                'train_nl_auroc':nl_auroc,
                 }
    
    def training_epoch_end(self, outputs):
        n_batch = len(outputs)
        loss = torch.tensor([x['loss'] for x in outputs]).mean()
                
        native_languages_acc = torch.tensor([x['train_native_languages_acc'] for x in outputs]).mean()
        nl_F1Score = torch.tensor([x['train_nl_F1score'] for x in outputs]).mean()
        #nl_confmatrix = torch.tensor([x['train_nl_Confmatrix'] for x in outputs]).mean()
        #nl_confmatrix = self.ConfutionMatrix_MultiClass_criterion.compute()
        #nl_auroc = torch.tensor([x['train_nl_auroc'] for x in outputs]).mean()

        self.log('epoch_loss' , loss, prog_bar=True, sync_dist=True)
        self.log('native-language accuracy',native_languages_acc, prog_bar=True, sync_dist=True)
        self.log('native-language F1Score',nl_F1Score, prog_bar=True, sync_dist=True)
        #self.log('nl_auroc',nl_auroc, prog_bar=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x, y_nl = batch       
        y_hat_nl = self(x)

        native_languages_loss =  self.cross_entropy_loss(y_hat_nl, y_nl)        
        loss = self.alpha * native_languages_loss
        
        native_languages_acc = self.accuracy(y_hat_nl.float(), y_nl)        
        
        return {'val_loss':loss, 
                'val_native_languages_acc':native_languages_acc,
        }


    def validation_epoch_end(self, outputs):
        n_batch = len(outputs)
        val_loss = torch.tensor([x['val_loss'] for x in outputs]).mean()
        
        native_languages_acc = torch.tensor([x['val_native_languages_acc'] for x in outputs]).mean()
        
        self.log('v_loss' , val_loss, prog_bar=True, sync_dist=True)
        self.log('v_nl_acc',native_languages_acc, prog_bar=True, sync_dist=True)
        
        
    def test_step(self, batch, batch_idx):       
        x, y_nl = batch
        y_hat_nl = self(x)

        native_languages_acc = self.accuracy(y_hat_nl.float(), y_nl)  
                    
        return {
                'test_native_languages_acc':native_languages_acc,
        }


    def test_epoch_end(self, outputs):
        n_batch = len(outputs)

        native_languages_acc = torch.tensor([x['test_native_languages_acc'] for x in outputs]).mean()        
        

        pbar = {'test_native_languages_acc':native_languages_acc.item(),
        } 

        self.logger.log_hyperparams(pbar)
        self.log_dict(pbar)
        

