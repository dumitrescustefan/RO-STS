import os, sys, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn import MSELoss
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from pytorch_lightning.callbacks import ModelCheckpoint

from tokenizers import Tokenizer


class STSRNNBaselineModel(pl.LightningModule):

    def __init__(self, vocab_size, lr=1e-04):
        super().__init__()

        # config params
        self.vocab_size = vocab_size
        self.embedding_dim = 256
        self.hidden_size = 512
        self.rnn_layers = 2
        self.lr = lr

        # architecture
        self.embeddings = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.rnn_layers,
            batch_first=True,
            dropout=0.1,
            bidirectional=True)
        self.mixer = nn.Linear(2 * 2 * self.hidden_size, 1) # 2 from bilstm, 2 from concating sentences
        self.loss = MSELoss()

        # data saving
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

    def _get_last_state(self, packed, lengths):
        sum_batch_sizes = torch.cat((
            torch.zeros(2, dtype=torch.int64),
            torch.cumsum(packed.batch_sizes, 0)
        ))
        sorted_lengths = lengths[packed.sorted_indices]
        last_seq_idxs = sum_batch_sizes[sorted_lengths] + torch.arange(lengths.size(0))
        last_seq_items = packed.data[last_seq_idxs]
        last_seq_items = last_seq_items[packed.unsorted_indices]
        return last_seq_items

    def forward(self, sentence1_batch, sentence1_lengths, sentence2_batch, sentence2_lengths):
        # embed sentence1 and 2: [
        sentence1_embeddings = self.embeddings(sentence1_batch) # [bs, max_seq_len1] -> [bs, max_seq_len1, emb_dim]
        sentence2_embeddings = self.embeddings(sentence2_batch) # [bs, max_seq_len2] -> [bs, max_seq_len2, emb_dim]
        #print("sentence1_embeddings {}".format(sentence1_embeddings.size()))

        # run sentence 1 through rnn, obtain a [bs, max_seq_len1, hidden_size]
        sentence1_packed = pack_padded_sequence(sentence1_embeddings, sentence1_lengths, batch_first=True, enforce_sorted=False)
        sentence1_output, sentence1_hidden = self.rnn(sentence1_packed) # hidden is zero at start
        #sentence1_output, _ = pad_packed_sequence(sentence1_output, batch_first=True) # [bs, max_seq_len1, 2*hidden_size]
        #print("sentence1_output {}".format(sentence1_output.size()))

        # run sentence 2 through rnn
        sentence2_packed = pack_padded_sequence(sentence2_embeddings, sentence2_lengths, batch_first=True, enforce_sorted=False)
        sentence2_output, sentence2_hidden = self.rnn(sentence2_packed)  # hidden is zero at start
        #sentence2_output, _ = pad_packed_sequence(sentence2_output, batch_first=True) # [bs, max_seq_len1, 2*hidden_size]

        # concat last states and send to mixer output layer
        sentence1_last_state = self._get_last_state(sentence1_output, sentence1_lengths) #entence1_output[:,-1,:] # [bs, 2*hidden_size]
        sentence2_last_state = self._get_last_state(sentence2_output, sentence2_lengths) #sentence2_output[:,-1,:] # [bs, 2*hidden_size]

        concat = torch.cat([sentence1_last_state, sentence2_last_state], dim=1) # [bs, 2*hidden_size]
        #print("concat {}".format(concat.size()))

        output = self.mixer(concat)  # [bs]

        return output

    def training_step(self, batch, batch_idx):
        similarity, sentence1_batch, sentence1_lengths, sentence2_batch, sentence2_lengths = batch
        output = self(sentence1_batch, sentence1_lengths, sentence2_batch, sentence2_lengths) # [bs]

        loss = self.loss(output.squeeze(), similarity)

        preds = output.squeeze()

        self.train_y_hat.extend(preds.detach().cpu().view(-1).numpy())
        self.train_y.extend(similarity.detach().cpu().view(-1).numpy())
        self.train_loss.append(loss.detach().cpu().numpy())

        # wandb logging
        #self.logger.experiment.log({'train/loss': loss})
        self.log('train/loss', loss)

        return loss#{"loss": loss}

    def training_epoch_end(self, outputs):
        pearson_score = pearsonr(self.train_y, self.train_y_hat)[0]
        spearman_score = spearmanr(self.train_y, self.train_y_hat)[0]

        self.log('train/pearson', pearson_score, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/spearman', spearman_score, on_step=False, on_epoch=True, prog_bar=True)

        self.train_y_hat = []
        self.train_y = []

        print() # print a new line to see individual epochs pbars

    def validation_step(self, batch, batch_idx):
        similarity, sentence1_batch, sentence1_lengths, sentence2_batch, sentence2_lengths = batch
        output = self(sentence1_batch, sentence1_lengths, sentence2_batch, sentence2_lengths)  # [bs]

        loss = self.loss(output.squeeze(), similarity)

        preds = output.squeeze()

        # log results 
        self.valid_y_hat.extend(preds.detach().cpu().view(-1).numpy())
        self.valid_y.extend(similarity.detach().cpu().view(-1).numpy())
        self.valid_loss.append(loss.detach().cpu().numpy())

        self.log('valid/loss', loss)

        return loss#{"loss": loss}

    def validation_epoch_end(self, outputs):
        pearson_score = pearsonr(self.valid_y, self.valid_y_hat)[0]
        spearman_score = spearmanr(self.valid_y, self.valid_y_hat)[0]

        self.log('valid/pearson', pearson_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('valid/spearman', spearman_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.valid_y_hat = []
        self.valid_y = []

    def test_step(self, batch, batch_idx):
        similarity, sentence1_batch, sentence1_lengths, sentence2_batch, sentence2_lengths = batch
        output = self(sentence1_batch, sentence1_lengths, sentence2_batch, sentence2_lengths)  # [bs]

        preds = output.squeeze()

        self.test_y_hat.extend(preds.detach().cpu().view(-1).numpy())
        self.test_y.extend(similarity.detach().cpu().view(-1).numpy())

        return loss #{"test_loss": loss}

    def test_epoch_end(self, outputs):
        pearson_score = pearsonr(self.test_y, self.test_y_hat)[0]
        spearman_score = spearmanr(self.test_y, self.test_y_hat)[0]

        self.log('test/pearson', pearson_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test/spearman', spearman_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.test_y_hat = []
        self.test_y = []

    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=self.lr)#, eps=1e-08)
