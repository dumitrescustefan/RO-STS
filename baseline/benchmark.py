import logging, os, sys, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn import MSELoss
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel, AutoConfig, Trainer, TrainingArguments
from pytorch_lightning.callbacks import EarlyStopping
from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr

from pytorch_lightning.loggers import WandbLogger
wandb_logger = WandbLogger()
from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    monitor='valid_loss',
    mode='min',
)

class STSBaselineModel (pl.LightningModule):

    def __init__(self, model_name="bert-base-uncased", lr=2e-05): #model_name="dumitrescustefan/bert-base-romanian-cased-v1")
        super().__init__()
        print("Loading AutoModel [{}]...".format(model_name))
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name, num_labels=1, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(0.2)
        self.mixer = nn.Linear(self.model.config.hidden_size, 1)
        self.loss_fct = MSELoss()
       
        self.lr = lr
        
        self.train_y_hat = []
        self.train_y = []
        self.train_loss = []
        self.valid_y_hat = []
        self.valid_y = []
        self.valid_loss = []
        self.test_y_hat = []
        self.test_y = []
        self.test_loss = []
  
        self.cnt = 0
        
    def forward(self, s, attn, sim):
        # we assume that AutoModel for outputs a tuple with  element [1] being the
        # pooler_output (torch.FloatTensor: of shape (batch_size, hidden_size)):
        if "bert" in self.model_name:
            output = self.model(input_ids=s, attention_mask=attn)
            pooled_sentence = self.dropout(output[1]) # [bs, hidden_size]
        elif "t5" in self.model_name:
            output = self.model.encoder(input_ids=s, attention_mask=attn, return_dict=True)
            pooled_sentence = output.last_hidden_state # [batch_size, seq_len, hidden_size]
            pooled_sentence = torch.mean(pooled_sentence, dim=1)
        logits = self.mixer(pooled_sentence) # [bs]
        loss = self.loss_fct(logits.squeeze(), sim)
        return loss, logits


    def training_step(self, batch, batch_idx):
        s, attn, sim = batch
        outputs = self(s, attn, sim)
        
        loss, logits = outputs[:2]
        preds = logits.squeeze()
        
        self.train_y_hat.extend(preds.detach().cpu().view(-1).numpy())
        self.train_y.extend(sim.detach().cpu().view(-1).numpy())
        self.train_loss.append(loss.detach().cpu().numpy())
        
        # wandb logging
        self.logger.experiment.log({'train/loss': loss})

        return {"loss": loss}
        
  
    def training_epoch_end(self, outputs):
        pearson_score = pearsonr(self.train_y, self.train_y_hat)[0]
        spearman_score = spearmanr(self.train_y, self.train_y_hat)[0]
        mean_train_loss = sum(self.train_loss)/len(self.train_loss)

        # wandb logging
        self.logger.experiment.log({"train/avg_loss": mean_train_loss})
        self.logger.experiment.log({"train/pearson": pearson_score})
        self.logger.experiment.log({"train/spearman": spearman_score})

        self.train_y_hat = []
        self.train_y = []
        self.train_loss = []

        
    def validation_step(self, batch, batch_idx):
        s, attn, sim = batch
        outputs = self(s, attn, sim)
        
        loss, logits = outputs[:2]
        preds = logits.squeeze()
        
        # log results 
        self.valid_y_hat.extend(preds.detach().cpu().view(-1).numpy())
        self.valid_y.extend(sim.detach().cpu().view(-1).numpy())
        self.valid_loss.append(loss.detach().cpu().numpy())
        self.logger.experiment.log({"valid/loss": loss})

        return {"valid_loss": loss}


    def validation_epoch_end(self, outputs):
        pearson_score = pearsonr(self.valid_y, self.valid_y_hat)[0]
        spearman_score = spearmanr(self.valid_y, self.valid_y_hat)[0]
        mean_val_loss = sum(self.valid_loss)/len(self.valid_loss)
        
        # wandb logging
        self.logger.experiment.log({"valid/avg_loss": mean_val_loss})
        self.logger.experiment.log({"valid/pearson": pearson_score})
        self.logger.experiment.log({"valid/spearman": spearman_score})

        #early stopping logging
        self.log_dict({"valid_loss": mean_val_loss,
                       "valid_pearson": pearson_score,
                       "valid_spearman": spearman_score})

        self.valid_y_hat = []
        self.valid_y = []
        self.valid_loss = []


    def test_step(self, batch, batch_idx):
        s, attn, sim = batch
        outputs = self(s, attn, sim)
        
        loss, logits = outputs[:2]
        preds = logits.squeeze()
        
        self.test_y_hat.extend(preds.detach().cpu().view(-1).numpy())
        self.test_y.extend(sim.detach().cpu().view(-1).numpy())
        self.test_loss.append(loss.detach().cpu().numpy()) # aici cu extend, si toate fara numpy

        # wandb logging
        self.logger.experiment.log({'test/loss': loss})
        return {"test_loss": loss}


    def test_epoch_end(self, outputs):
        pearson_score = pearsonr(self.test_y, self.test_y_hat)[0]
        spearman_score = spearmanr(self.test_y, self.test_y_hat)[0]
        mean_test_loss = sum(self.test_loss)/len(self.test_loss)

        # wandb logging
        self.logger.experiment.log({"test/avg_loss": mean_test_loss})
        self.logger.experiment.log({"test/pearson": pearson_score})
        self.logger.experiment.log({"test/spearman": spearman_score})
      
        self.test_y_hat = []
        self.test_y = []
        self.test_loss = []

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
 

    max_seq_len_s1, max_seq_len_s2, max_seq_len, batch_size = 0, 0, 0, len(batch)

    for example in batch:
        max_seq_len_s1 = max(max_seq_len_s1, len(example["sentence1"]))
        max_seq_len_s2 = max(max_seq_len_s2, len(example["sentence2"]))
        max_seq_len = max(max_seq_len, max_seq_len_s1+max_seq_len_s2)

    s = []
    attn = []
    sim = []
    for i, example in enumerate(batch):
        ten =  torch.tensor( example["sentence1"] + example["sentence2"] + [model.tokenizer.pad_token_id] * (max_seq_len - len(example["sentence1"]) - len(example["sentence2"])), dtype=torch.long)
        s.append(ten)
        attn.append((ten>0.0).float())
        sim.append(example["sim"])

    s = torch.stack(s, dim=0)
    attn = torch.stack(attn, dim=0)
    sim = torch.tensor(sim, dtype=torch.float)

    return s, attn, sim

batch_size = 16

#train_dataset = MyDataset(tokenizer=model.tokenizer, file_path="../ro-sts/train.tsv", block_size=512)
#val_dataset = MyDataset(tokenizer=model.tokenizer, file_path="../ro-sts/dev.tsv", block_size=512)
#test_dataset = MyDataset(tokenizer=model.tokenizer, file_path="../ro-sts/test.tsv", block_size=512)

train_dataset = MyDataset(tokenizer=model.tokenizer, file_path="./ro-sts/sts-train.csv", block_size=512)
val_dataset = MyDataset(tokenizer=model.tokenizer, file_path="./ro-sts/sts-dev.csv", block_size=512)
test_dataset = MyDataset(tokenizer=model.tokenizer, file_path="./ro-sts/sts-test.csv", block_size=512)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True, collate_fn=my_collate, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False, collate_fn=my_collate, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False, collate_fn=my_collate, pin_memory=True)


print("Train dataset has {} instances, meaning {:.0f} steps.".format(len(train_dataset), len(train_dataset)/batch_size))
print("Valid dataset has {} instances, meaning {:.0f} steps.".format(len(val_dataset), len(val_dataset)/batch_size))


early_stop = EarlyStopping(
    monitor='valid_pearson',
    patience=3,
    strict=False,
    verbose=False,
    mode='min'
)


trainer = pl.Trainer(
    gpus=1,
    checkpoint_callback=checkpoint_callback,
    callbacks=[early_stop],
    #limit_train_batches=20,
    #limit_val_batches=10,
    accumulate_grad_batches=8,
    weights_save_path='model',
    gradient_clip_val=1.0,
    logger=wandb_logger,
    auto_lr_find=False,
    #progress_bar_refresh_rate=10,
)

trainer.fit(model, train_dataloader, val_dataloader)

#trainer.test(model, test_dataloader)

