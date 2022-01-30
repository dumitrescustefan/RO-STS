import logging, os, sys, json, torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn import MSELoss
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel, AutoConfig, Trainer, TrainingArguments
from pytorch_lightning.callbacks import EarlyStopping
from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr

from pytorch_lightning.callbacks import ModelCheckpoint

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TransformerModel (pl.LightningModule):
    def __init__(self, model_name="dumitrescustefan/bert-base-romanian-cased-v1", lr=2e-05, model_max_length=512):
        super().__init__()
        print("Loading AutoModel [{}]...".format(model_name))
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name, num_labels=1, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(0.2)

        self.loss_fct = MSELoss()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.lr = lr
        self.model_max_length = model_max_length
        
        self.train_y_hat = []
        self.train_y = []
        self.train_loss = []
        self.valid_y_hat = []
        self.valid_y = []
        self.valid_loss = []
        self.test_y_hat = []
        self.test_y = []
        self.test_loss = []

        # add pad token
        self.validate_pad_token()
    
    def validate_pad_token(self):
        if self.tokenizer.pad_token is not None:
            return
        if self.tokenizer.sep_token is not None:
            print(f"\tNo PAD token detected, automatically assigning the SEP token as PAD.")
            self.tokenizer.pad_token = self.tokenizer.sep_token
            return
        if self.tokenizer.eos_token is not None:
            print(f"\tNo PAD token detected, automatically assigning the EOS token as PAD.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            return
        if self.tokenizer.bos_token is not None:
            print(f"\tNo PAD token detected, automatically assigning the BOS token as PAD.")
            self.tokenizer.pad_token = self.tokenizer.bos_token
            return
        if self.tokenizer.cls_token is not None:
            print(f"\tNo PAD token detected, automatically assigning the CLS token as PAD.")
            self.tokenizer.pad_token = self.tokenizer.cls_token
            return
        raise Exception("Could not detect SEP/EOS/BOS/CLS tokens, and thus could not assign a PAD token which is required.")
        
         
        
    def forward(self, s1, s2, sim):
        o1 = self.model(input_ids=s1["input_ids"].to(self.device), attention_mask=s1["attention_mask"].to(self.device), return_dict=True)
        o2 = self.model(input_ids=s2["input_ids"].to(self.device), attention_mask=s2["attention_mask"].to(self.device), return_dict=True)
        pooled_sentence1 = o1.last_hidden_state # [batch_size, seq_len, hidden_size]
        pooled_sentence1 = torch.mean(pooled_sentence1, dim=1) # [batch_size, hidden_size]
        pooled_sentence2 = o2.last_hidden_state  # [batch_size, seq_len, hidden_size]
        pooled_sentence2 = torch.mean(pooled_sentence2, dim=1) # [batch_size, hidden_size]

        cosines = self.cos(pooled_sentence1, pooled_sentence2).squeeze() # [batch_size]
        loss = self.loss_fct(cosines, sim)
        return loss, cosines

    def training_step(self, batch, batch_idx):
        s1, s2, sim = batch
        
        loss, predicted_sims = self(s1, s2, sim)

        self.train_y_hat.extend(predicted_sims.detach().cpu().view(-1).numpy())
        self.train_y.extend(sim.detach().cpu().view(-1).numpy())
        self.train_loss.append(loss.detach().cpu().numpy())

        return {"loss": loss}

    def training_epoch_end(self, outputs):
        pearson_score = pearsonr(self.train_y, self.train_y_hat)[0]
        spearman_score = spearmanr(self.train_y, self.train_y_hat)[0]
        mean_train_loss = sum(self.train_loss)/len(self.train_loss)

        self.log("train/avg_loss", mean_train_loss, prog_bar=True)
        self.log("train/pearson", pearson_score, prog_bar=False)
        self.log("train/spearman", spearman_score, prog_bar=False)

        self.train_y_hat = []
        self.train_y = []
        self.train_loss = []

    def validation_step(self, batch, batch_idx):
        s1, s2, sim = batch
        
        loss, predicted_sims = self(s1, s2, sim)

        self.valid_y_hat.extend(predicted_sims.detach().cpu().view(-1).numpy())
        self.valid_y.extend(sim.detach().cpu().view(-1).numpy())
        self.valid_loss.append(loss.detach().cpu().numpy())

        return {"loss": loss}


    def validation_epoch_end(self, outputs):
        pearson_score = pearsonr(self.valid_y, self.valid_y_hat)[0]
        spearman_score = spearmanr(self.valid_y, self.valid_y_hat)[0]
        mean_val_loss = sum(self.valid_loss)/len(self.valid_loss)
        
        self.log("valid/avg_loss", mean_val_loss, prog_bar=True)
        self.log("valid/pearson", pearson_score, prog_bar=True)
        self.log("valid/spearman", spearman_score, prog_bar=True)

        self.valid_y_hat = []
        self.valid_y = []
        self.valid_loss = []


    def test_step(self, batch, batch_idx):
        s1, s2, sim = batch
        
        loss, predicted_sims = self(s1, s2, sim)

        self.test_y_hat.extend(predicted_sims.detach().cpu().view(-1).numpy())
        self.test_y.extend(sim.detach().cpu().view(-1).numpy())
        self.test_loss.append(loss.detach().cpu().numpy())

        return {"loss": loss}


    def test_epoch_end(self, outputs):
        pearson_score = pearsonr(self.test_y, self.test_y_hat)[0]
        spearman_score = spearmanr(self.test_y, self.test_y_hat)[0]
        mean_test_loss = sum(self.test_loss)/len(self.test_loss)

        self.log("test/avg_loss", mean_test_loss, prog_bar=True)
        self.log("test/pearson", pearson_score, prog_bar=True)
        self.log("test/spearman", spearman_score, prog_bar=True)
      
        self.test_y_hat = []
        self.test_y = []
        self.test_loss = []

    def configure_optimizers(self):
        return torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=self.lr, eps=1e-08)


class MyDataset(Dataset):
    def __init__(self, tokenizer, file_path: str):
        self.file_path = file_path
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
            if len(parts) != 3:
                print(".")
                continue
            sim = parts[0]
            sentence1 = parts[1]
            sentence2 = parts[2]
            instance = {
                "sentence1": sentence1,
                "sentence2": sentence2,
                "sim": float(sim)/5.
            }
            self.instances.append(instance)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        return self.instances[i] #torch.tensor([0], dtype=torch.long)


def my_collate(batch):
    # batch is a list of batch_size number of instances; each instance is a dict, as given by MyDataset.__getitem__()
    # return is a sentence1_batch, sentence2_batch, sims
    # the first two return values are dynamic batching for sentences 1 and 2, and [bs] is the sims for each of them
    # sentence1_batch is a dict like:
    """
    'input_ids': tensor([[101, 8667, 146, 112, 182, 170, 1423, 5650, 102],
                         [101, 1262, 1330, 5650, 102, 0, 0, 0, 0],
                         [101, 1262, 1103, 1304, 1304, 1314, 1141, 102, 0]]),
    'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0]]),
    'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1],
    """
    sentence1_batch = []
    sentence2_batch = []
    sims = []
    for instance in batch:
        #print(instance["sentence1"])
        sentence1_batch.append(instance["sentence1"])
        sentence2_batch.append(instance["sentence2"])
        sims.append(instance["sim"])

    sentence1_batch = model.tokenizer(sentence1_batch, padding=True, max_length = model.model_max_length, truncation=True, return_tensors="pt")
    sentence2_batch = model.tokenizer(sentence2_batch, padding=True, max_length = model.model_max_length, truncation=True, return_tensors="pt")
    sims = torch.tensor(sims, dtype=torch.float)

    return sentence1_batch, sentence2_batch, sims

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--accumulate_grad_batches', type=int, default=16)
    parser.add_argument('--model_name', type=str, default="dumitrescustefan/bert-base-romanian-cased-v1") #xlm-roberta-base
    parser.add_argument('--lr', type=float, default=2e-05)
    parser.add_argument('--model_max_length', type=int, default=512)
    parser.add_argument('--experiment_iterations', type=int, default=1)
    args = parser.parse_args()
    
    
    print("Batch size is {}, accumulate grad batches is {}, final batch_size is {}\n".format(args.batch_size, args.accumulate_grad_batches, args.batch_size * args.accumulate_grad_batches))
    
    model = TransformerModel(model_name=args.model_name, lr=args.lr, model_max_length=args.model_max_length) # need to load for tokenizer
    
    print("Loading data...") 
    train_dataset = MyDataset(tokenizer=model.tokenizer, file_path="../dataset/text-similarity/RO-STS.train.tsv")
    val_dataset = MyDataset(tokenizer=model.tokenizer, file_path="../dataset/text-similarity/RO-STS.dev.tsv")
    test_dataset = MyDataset(tokenizer=model.tokenizer, file_path="../dataset/text-similarity/RO-STS.test.tsv")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, collate_fn=my_collate, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, collate_fn=my_collate, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, collate_fn=my_collate, pin_memory=True)


    print("Train dataset has {} instances.".format(len(train_dataset)))
    print("Valid dataset has {} instances.".format(len(val_dataset)))
    print("Test dataset has {} instances.\n".format(len(test_dataset)))

    itt = 0
    
    v_p = []
    v_s = []
    v_l = []
    t_p = []
    t_s = []
    t_l = []
    while itt<args.experiment_iterations:
        print("Running experiment {}/{}".format(itt+1, args.experiment_iterations))
        
        model = TransformerModel(model_name=args.model_name, lr=args.lr, model_max_length=args.model_max_length)
        
        early_stop = EarlyStopping(
            monitor='valid/pearson',
            patience=4,
            verbose=True,
            mode='max'
        )
        
        trainer = pl.Trainer(
            gpus=args.gpus,
            callbacks=[early_stop],
            #limit_train_batches=5,
            #limit_val_batches=2,
            accumulate_grad_batches=args.accumulate_grad_batches,
            gradient_clip_val=1.0,
            enable_checkpointing=False
        )
        trainer.fit(model, train_dataloader, val_dataloader)

        resultd = trainer.test(model, val_dataloader)
        result = trainer.test(model, test_dataloader)

        with open("results_{}_of_{}.json".format(itt+1, args.experiment_iterations),"w") as f:
            json.dump(result[0], f, indent=4, sort_keys=True)

        v_p.append(resultd[0]['test/pearson'])
        v_s.append(resultd[0]['test/spearman'])
        v_l.append(resultd[0]['test/avg_loss'])
        t_p.append(result[0]['test/pearson'])
        t_s.append(result[0]['test/spearman'])
        t_l.append(result[0]['test/avg_loss'])
        
        itt += 1


    print("Done, writing results...")
    result = {}
    result["valid_pearson"] = sum(v_p)/args.experiment_iterations
    result["valid_spearman"] = sum(v_s)/args.experiment_iterations
    result["valid_loss"] = sum(v_l)/args.experiment_iterations
    result["test_pearson"] = sum(t_p)/args.experiment_iterations
    result["test_spearman"] = sum(t_s)/args.experiment_iterations
    result["test_loss"] = sum(t_l)/args.experiment_iterations

    with open("results_of_{}.json".format(args.model_name.replace("/", "_")), "w") as f:
        json.dump(result, f, indent=4, sort_keys=True)
    
    from pprint import pprint
    pprint(result)