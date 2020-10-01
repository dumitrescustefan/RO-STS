import logging, os, sys, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from pytorch_lightning.callbacks import EarlyStopping
from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr

from pytorch_lightning.loggers import WandbLogger
wandb_logger = WandbLogger()
from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
)

class STSBaselineModel (pl.LightningModule):
    def __init__(self, model_name="t5-base", lr=2e-05): #model_name="dumitrescustefan/bert-base-romanian-cased-v1"):
        super().__init__()
        print("Loading AutoModel [{}]...".format(model_name))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.2)
        self.mixer = nn.Linear(self.encoder.config.hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()
       
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

       
        self.lr = lr
        
        self.train_y_hat = []
        self.train_y = []
        self.train_loss = []
        self.valid_y_hat = []
        self.valid_y = []
        self.valid_loss = []
  
        self.cnt = 0
        
    def forward(self, s1, s2):
        output1 = self.encoder(s1, return_dict=True)
        output2 = self.encoder(s2, return_dict=True)
        # we assume that AutoModel outputs a tuple with  element [1] being the
        # pooler_output (torch.FloatTensor: of shape (batch_size, hidden_size)):

        #pooled_sentence1 = self.dropout(output1[1]) # [bs, hidden_size]
        #pooled_sentence2 = self.dropout(output2[1]) # [bs, hidden_size]
        pooled_sentence1 = torch.mean(output1.last_hidden_state, dim=1)
        pooled_sentence2 = torch.mean(output2.last_hidden_state, dim=1)
        #print(pooled_sentence1.size())

        #mixer_input = torch.cat([pooled_sentence1, pooled_sentence2] , dim=1) # [bs, hidden * 2]

        #return self.sigmoid(self.mixer(mixer_input)).squeeze() # [bs]
        return self.cos(pooled_sentence1, pooled_sentence2)

    def training_step(self, batch, batch_idx):
        s1, s2, y = batch
        y_hat = self(s1, s2)
        
        loss = F.mse_loss(y_hat.view(-1), y.view(-1))        
        
        # log results 
        self.train_y_hat.extend(y_hat.detach().cpu().view(-1).numpy())
        self.train_y.extend(y.detach().cpu().view(-1).numpy())
        self.train_loss.append(loss.detach().cpu().numpy())
        
        return pl.TrainResult(loss)
        
    def training_epoch_end(self, outputs):
        result = pl.EvalResult()   
        result.log('train/lss', sum(self.train_loss)/len(self.train_loss), prog_bar=False, on_step=False, on_epoch=True)
        result.log("train/pearson", pearsonr(self.train_y, self.train_y_hat)[0] , prog_bar=True, on_step=False, on_epoch=True)
        result.log("train/spearman", spearmanr(self.train_y, self.train_y_hat)[0] , prog_bar=False, on_step=False, on_epoch=True)
        self.train_y_hat = []
        self.train_y = []
        self.train_loss = []
            
        return result 
    
    def validation_step(self, batch, batch_idx):
        s1, s2, y = batch
        y_hat = self(s1, s2)
        
        loss = F.mse_loss(y_hat.view(-1), y.view(-1))
        
        # log results 
        self.valid_y_hat.extend(y_hat.detach().cpu().view(-1).numpy())
        self.valid_y.extend(y.detach().cpu().view(-1).numpy())
        self.valid_loss.append(loss.detach().cpu().numpy())
        
        return pl.EvalResult(loss)

    def validation_epoch_end(self, outputs):
        pearson_score = pearsonr(self.valid_y, self.valid_y_hat)[0]
        
        result = pl.EvalResult()
        result.log('valid/lss', sum(self.valid_loss)/len(self.valid_loss), prog_bar=True, on_step=False, on_epoch=True)
        result.log("valid/pearson", pearson_score , prog_bar=True, on_step=False, on_epoch=True)
        result.log("valid/spearman", spearmanr(self.valid_y, self.valid_y_hat)[0] , prog_bar=False, on_step=False, on_epoch=True)
        
        self.valid_y_hat = []
        self.valid_y = []
        self.valid_loss = []
        
        return result

    def test_step(self, batch, batch_idx):
        result = self.validation_step(batch, batch_idx)
        results.rename_keys({"val_loss": "test_loss", "val_pearson_score": "test_pearson_score", \
                             "val_spearman_score": "test_spearman_score"})
        return result

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in [outputs]]).mean()
        pearson_score = self.pearsonr.get_score()
        spearnman_score = self.spearmanr.get_score()
        result = pl.EvalResult(checkpoint_on=avg_loss, early_stop_on=pearson_score)
        result.log_dict({"test_avg_loss": avg_loss, "test_pearson_score": pearson_score, \
                   "test_spearman_score": spearman_score})
        return result

    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=self.lr, eps=1e-08)


class MyDataset(Dataset):
    def __init__(self, tokenizer, file_path: str, block_size: int):
        self.file_path = file_path
        self.ncols = block_size
        self.instances = []
        print("Reading corpus: {}".format(file_path))

        # checks
        assert os.path.isfile(file_path)
        with open(file_path, "r", encoding="utf8") as f:
            lines = f.readlines()
        for line in lines:
            if line.strip()=="":
                break
            #print(line.split("\t"))
            #_, sim, sentence1, sentence2 = line.strip().split("\t")
            parts = line.strip().split("\t")
            sim = parts[4]
            sentence1 = parts[5]
            sentence2 = parts[6]
            instance = {
                "sentence1": tokenizer.encode(sentence1.strip(), add_special_tokens=True),
                "sentence2": tokenizer.encode(sentence2.strip(), add_special_tokens=True),
                "sim": float(sim)/2.5 - 1.
            }
            self.instances.append(instance)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i) -> torch.Tensor:
        return self.instances[i] #torch.tensor([0], dtype=torch.long)


model = STSBaselineModel()

def my_collate(batch):
    # batch is a list of batch_size number of instances; each instance is a dict, as given by MyDataset.__getitem__()
    # return is a [bs, max_seq_len_s1],  [bs, max_seq_len_s2], [bs]
    # the first two return values are dynamic batching for sentences 1 and 2, and [bs] is the sims for each of them
 
    max_seq_len_s1, max_seq_len_s2, batch_size = 0, 0, len(batch)
    for example in batch:
        max_seq_len_s1 = max(max_seq_len_s1, len(example["sentence1"]))
        max_seq_len_s2 = max(max_seq_len_s2, len(example["sentence2"]))

    s1 = []
    s2 = []
    sim = []
    for i, example in enumerate(batch):
        #print( example["sentence1"]+[0]*(max_seq_len_s1-len(example["sentence1"])) )
        s1.append( torch.tensor(example["sentence1"] + [model.tokenizer.pad_token_id] * (max_seq_len_s1 - len(example["sentence1"])), dtype=torch.long) )
        s2.append( torch.tensor(example["sentence2"] + [model.tokenizer.pad_token_id] * (max_seq_len_s2 - len(example["sentence2"])), dtype=torch.long) )
        sim.append(example["sim"])
    s1 = torch.stack(s1, dim=0)
    s2 = torch.stack(s2, dim=0)
    sim = torch.tensor( sim, dtype=torch.float )
    #print()
    #print(example)
    #print(s1)
    return s1, s2, sim


batch_size = 16

#train_dataset = MyDataset(tokenizer=model.tokenizer, file_path="../ro-sts/train.tsv", block_size=512)
#val_dataset = MyDataset(tokenizer=model.tokenizer, file_path="../ro-sts/dev.tsv", block_size=512)
#test_dataset = MyDataset(tokenizer=model.tokenizer, file_path="../ro-sts/test.tsv", block_size=512)

train_dataset = MyDataset(tokenizer=model.tokenizer, file_path="../ro-sts/sts-train.csv", block_size=512)
val_dataset = MyDataset(tokenizer=model.tokenizer, file_path="../ro-sts/sts-dev.csv", block_size=512)
test_dataset = MyDataset(tokenizer=model.tokenizer, file_path="../ro-sts/sts-test.csv", block_size=512)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True, collate_fn=my_collate, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False, collate_fn=my_collate, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False, collate_fn=my_collate, pin_memory=True)

print("Train dataset has {} instances, meaning {:.0f} steps.".format(len(train_dataset), len(train_dataset)/batch_size))
print("Valid dataset has {} instances, meaning {:.0f} steps.".format(len(val_dataset), len(val_dataset)/batch_size))

#early_stop = EarlyStopping(
#    monitor='val_pearson_score',
#    patience=3,
#    strict=False,
#    verbose=True,
#    mode='min'
#)

trainer = pl.Trainer(
    gpus=1,
    checkpoint_callback=checkpoint_callback,
    #early_stop_callback=early_stop,
    #limit_train_batches=20,
    #limit_val_batches=10,
    accumulate_grad_batches=1,
    weights_save_path='model',
    gradient_clip_val=1.0,
    logger=wandb_logger,
    auto_lr_find=False,
    #progress_bar_refresh_rate=10,
)

trainer.fit(model, train_dataloader, val_dataloader)

#trainer.test(model, test_dataloader)