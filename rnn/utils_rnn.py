import os, sys

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import Whitespace, Digits
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.trainers import WordPieceTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_tokenizer(train_dev_test_files, vocab_size):
    """
    If a tokenizer does not exist, it will read input data and train a tokenizer, placing it in rnn_files
    Otherwise, it returns the trained tokenizer
    """

    def _train_tokenizer(train_dev_test_files):
        """
        This function will train a new tokenizer.
        """
        if not os.path.exists("rnn_files"):
            os.makedirs("rnn_files")

        # extract data from files
        with open(os.path.join("rnn_files","data.raw"), "w", encoding="utf8") as f:
            for file_path in train_dev_test_files:
                with open(file_path, "r", encoding="utf8") as r:
                    lines = r.readlines()
                for line in lines:
                    if line.strip() == "":
                        break
                    parts = line.strip().split("\t")
                    sentence1 = parts[5]
                    sentence2 = parts[6]
                    f.write(sentence1 + "\n")
                    f.write(sentence2 + "\n")

        # train tokenizer
        tokenizer = Tokenizer(WordPiece())

        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits(individual_digits=True)])
        tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
        from tokenizers.processors import TemplateProcessing

        tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", 1),
                ("[SEP]", 2),
            ],
        )

        trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

        tokenizer.train(trainer, [os.path.join("rnn_files","data.raw")])

        # save
        model_files = tokenizer.model.save("rnn_files", "tokenizer")
        tokenizer.model = WordPiece.from_file(*model_files, unk_token="[UNK]")

        tokenizer.save(os.path.join("rnn_files","tokenizer.json"))

    # train tokenizer if not already trained
    if not os.path.exists(os.path.join("rnn_files","tokenizer.json")):
        print("Tokenizer not found, training from scratch ...")
        _train_tokenizer()

    print("Loading tokenizer ...")
    tokenizer = Tokenizer.from_file(os.path.join("rnn_files","tokenizer.json"))

    return tokenizer

class STSRNNDataset(Dataset):
    def __init__(self, tokenizer, file_path: str):
        self.file_path = file_path
        self.instances = []
        print("Reading corpus: {}".format(file_path))

        # checks
        assert os.path.isfile(file_path)
        with open(file_path, "r", encoding="utf8") as f:
            lines = f.readlines()
        for line in lines:
            if line.strip() == "":
                break
            parts = line.strip().split("\t")
            sim = parts[4]
            sentence1 = parts[5]
            sentence2 = parts[6]
            instance = {
                "sentence1": tokenizer.encode(sentence1.strip(), add_special_tokens=True).ids,
                "sentence2": tokenizer.encode(sentence2.strip(), add_special_tokens=True).ids,
                "sim": float(sim) / 2.5 - 1.
            }
            self.instances.append(instance)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i) -> torch.Tensor:
        return self.instances[i]


def get_dataloaders(train_file, dev_file, test_file, tokenizer, batch_size):
    def my_collate(batch):
        # batch is a list of batch_size number of instances; each instance is a dict, as given by MyDataset.__getitem__()

        max_seq_len_s1, max_seq_len_s2, batch_size = 0, 0, len(batch)

        for example in batch:
            max_seq_len_s1 = max(max_seq_len_s1, len(example["sentence1"]))
            max_seq_len_s2 = max(max_seq_len_s2, len(example["sentence2"]))

        sentence1_batch = []
        sentence2_batch = []
        sentence1_lengths = []
        sentence2_lengths = []
        similarity = []

        for i, instance in enumerate(batch):
            # pad sentence1
            sentence1_batch.append(
                    torch.tensor(instance["sentence1"] + [0] * (max_seq_len_s1 - len(instance["sentence1"])), dtype=torch.long)
                )
            sentence1_lengths.append(len(instance["sentence1"]))
            # pad sentence2
            sentence2_batch.append(
                    torch.tensor(instance["sentence2"] + [0] * (max_seq_len_s2 - len(instance["sentence2"])) ,dtype=torch.long)
                )
            sentence2_lengths.append(len(instance["sentence2"]))
            # add similarity score
            similarity.append(instance["sim"])

        sentence1_batch = torch.stack(sentence1_batch, dim=0)
        sentence1_lengths = torch.tensor(sentence1_lengths, dtype=torch.long)
        sentence2_batch = torch.stack(sentence2_batch, dim=0)
        sentence2_lengths = torch.tensor(sentence2_lengths, dtype=torch.long)
        similarity = torch.tensor(similarity, dtype=torch.float)

        return similarity, sentence1_batch, sentence1_lengths, sentence2_batch, sentence2_lengths

    train_dataset = STSRNNDataset(tokenizer=tokenizer, file_path="../ro-sts/sts-train.csv")
    val_dataset = STSRNNDataset(tokenizer=tokenizer, file_path="../ro-sts/sts-dev.csv")
    test_dataset = STSRNNDataset(tokenizer=tokenizer, file_path="../ro-sts/sts-test.csv")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True,
                                  collate_fn=my_collate,
                                  pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False, collate_fn=my_collate,
                                pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False,
                                 collate_fn=my_collate,
                                 pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader


import os, sys, math

sys.path.insert(0, '../../..')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Attention(nn.Module):
    def __init__(self, encoder_size, decoder_size, type="additive"):
        """ Attention module.
            Args:
                encoder_size (int): Size of the encoder's output (as input for the decoder).
                decoder_size (int): Size of the decoder's output.
                type (string): One of several types of attention

            See: https://arxiv.org/pdf/1902.02181.pdf

            Notes:
                Self-Attention(Intra-attention) Relating different positions of the same input sequence. Theoretically the self-attention can adopt any score functions above, but just replace the target sequence with the same input sequence.
                Global/Soft	Attending to the entire input state space.
                Local/Hard	Attending to the part of input state space; i.e. a patch of the input image.
        """
        super(Attention, self).__init__()
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        self.type = type

        # transforms encoder states into keys
        self.key_annotation_function = nn.Linear(self.encoder_size, self.encoder_size, bias=False)
        # transforms encoder states into values
        self.value_annotation_function = nn.Linear(self.encoder_size, self.encoder_size, bias=False)
        # transforms the hidden state into query
        self.query_annotation_function = nn.Linear(self.decoder_size, self.encoder_size,
                                                   bias=False)  # NOTE: transforming q to K size

        if type == "additive":
            # f(q, K) = wimp*tanh(W1K + W2q + b) , Bahdanau et al., 2015
            self.V = nn.Linear(self.encoder_size, 1, bias=False)
            self.W1 = nn.Linear(self.encoder_size, self.encoder_size, bias=False)
            self.W2 = nn.Linear(self.encoder_size, self.encoder_size,
                                bias=False)  # encoder size because q is now K's size, otherwise dec_size to enc_size
            self.b = nn.Parameter(torch.zeros(self.encoder_size))

        elif type == "coverage":  # https://arxiv.org/pdf/1601.04811.pdf
            # f(q, K) = wimp*tanh(W1K + W2q + b) , Bahdanau et al., 2015
            self.V = nn.Linear(self.encoder_size, 1, bias=False)
            self.W1 = nn.Linear(self.encoder_size, self.encoder_size, bias=False)
            self.W2 = nn.Linear(self.encoder_size, self.encoder_size,
                                bias=False)  # encoder size because q is now K's size, otherwise dec_size to enc_size
            self.b = nn.Parameter(torch.zeros(self.encoder_size))

            self.coverage_dim = 10
            self.coverage_input_size = self.coverage_dim + 1 + self.encoder_size + self.encoder_size
            self.cov_gru = nn.GRU(self.coverage_input_size, self.coverage_dim, batch_first=True)
            self.W3 = nn.Linear(self.coverage_dim, self.encoder_size)

        elif (type == "multiplicative" or type == "dot"):
            # f(q, K) = q^t K , Luong et al., 2015
            # direct dot product, nothing to declare here
            pass

        elif type == "scaled multiplicative" or type == "scaled dot":
            # f(q, K) = multiplicative / sqrt(dk) , Vaswani et al., 2017
            self.scale = math.sqrt(self.encoder_size)

        elif type == "general" or type == "bilinear":
            # f(q, K) = q^t WK , Luong et al., 2015
            self.W = nn.Linear(self.encoder_size, self.encoder_size, bias=False)

        elif type == "biased general":
            # f(q, K) = K|(W q + b) Sordoni et al., 2016
            pass
        elif type == "activated general":
            # f(q, K) = act(q|WK + b) Ma et al., 2017
            pass
        elif type == "concat":
            # f(q, K) = act(W[K;q] + b) , Luong et al., 2015
            pass
        elif type == "p":
            # https://arxiv.org/pdf/1702.04521.pdf pagina 3, de adaugat predict-ul in attention, KVP si Q
            pass
        else:
            raise Exception("Attention type not properly defined! (got type={})".format(self.type))

    def _reshape_state_h(self, state_h):
        """
        Reshapes the hidden state to desired shape
        Input: [num_layers * 1, batch_size, decoder_hidden_size]
        Output: [batch_size, 1, decoder_hidden_state]
        Args:
            state_h (tensor): Hidden state of the decoder.
                [num_layers * 1, batch_size, decoder_hidden_size]

        Returns:
            The reshaped hidden state.
                [batch_size, 1, decoder_hidden_state]
        """
        num_layers, batch_size, hidden_size = state_h.size()
        # in case the decoder has more than 1 layer, take only the last one -> [1, batch_size, decoder_hidden_size]
        if num_layers > 1:
            state_h = state_h[num_layers - 1:num_layers, :, :]

        # [1, batch_size, decoder_hidden_size] -> [batch_size, 1, decoder_hidden_size]
        return state_h.permute(1, 0, 2)

    def init_batch(self, batch_size, enc_seq_len):
        if self.type == "coverage":
            self.C = torch.zeros(batch_size, enc_seq_len, self.coverage_dim)
            self.gru_input = torch.zeros(batch_size, 1, self.coverage_input_size)

    def _coverage_compute_next_C(self, attention_weights, enc_output, state_h):
        # attention_weights:        [batch_size, seq_len, 1]
        # enc_output :              [batch_size, seq_len, encoder_size]
        # state_h (after reshape):  [batch_size, 1, encoder_size]
        # self.C at prev timestep:  [batch_size, seq_len, coverage_dim]
        seq_len = enc_output.size[1]
        for i in range(seq_len):
            self.gru_input[:, :, 0:self.coverage_dim] = self.C[:, i:i + 1,
                                                        :]  # cat self.C at prev timestep for position i of source word
            self.gru_input[:, :, self.coverage_dim:self.coverage_dim + 1] = attention_weights[:, i:i + 1,
                                                                            :]  # cat attention weight
            self.gru_input[:, :, self.coverage_dim + 1:self.coverage_dim + 1 + self.encoder_size] = enc_output[:,
                                                                                                    i:i + 1,
                                                                                                    :]  # cat encoder output
            self.gru_input[:, :, self.coverage_dim + 1 + self.encoder_size:] = state_h[:, 0:1, :]  # cat state_h
            self.C[:, i:i + 1, :] = self.cov_gru(self.gru_input)

    def _energy(self, K, Q):
        """
            Calculates the compatibility function f(query, keys)

            Args:
                K (tensor): Keys tensor of size [batch_size, seq_len, encoder_size]
                Q (tensor): Query tensor of size [batch_size, 1, decoder_size], but now dec_size is enc_size due to Q annotation

            Returns:
                energy tensor of size [batch_size, seq_len, 1]
        """
        if self.type == "additive":
            return self.V(torch.tanh(self.W1(K) + self.W2(Q) + self.b))

        elif self.type == "coverage":
            return self.V(torch.tanh(self.W1(K) + self.W2(Q) + self.W3(self.C) + self.b))

        elif self.type == "multiplicative" or self.type == "dot":
            # q^t K means batch matrix multiplying K with q transposed:
            # bmm( [batch_size, seq_len, enc_size] , [batch_size, enc_size, 1] ) -> [batch_size, seq_len, 1]
            return torch.bmm(K, Q.transpose(1, 2))

        elif self.type == "scaled multiplicative" or self.type == "scaled dot":
            # same as multiplicative but scaled
            return torch.bmm(K, Q.transpose(1, 2)) / self.scale

        elif self.type == "general" or self.type == "bilinear":
            # f(q, K) = q^t WK , Luong et al., 2015
            return torch.bmm(self.W(K), Q.transpose(1, 2))

    def forward(self, enc_output, state_h, mask=None):
        """
        This function calculates the context vector of the attention layer, given the hidden state and the encoder
        last lstm layer output.
        Args:
            state_h (tensor): The raw hidden state of the decoder's LSTM
                Shape: [num_layers * 1, batch_size, decoder_size].
            enc_output (tensor): The output of the last LSTM encoder layer.
                Shape: [batch_size, seq_len, encoder_size].
            mask (tensor): 1 and 0 as for encoder input
                Shape: [batch_size, seq_len].
        Returns:
            context (tensor): The context vector. Shape: [batch_size, encoder_size]
            attention_weights (tensor): Attention weights. Shape: [batch_size, seq_len, 1]
        """
        batch_size = enc_output.shape[0]
        seq_len = enc_output.shape[1]
        state_h = self._reshape_state_h(state_h)  # [batch_size, 1, decoder_size]

        # get K, V, Q
        K = self.key_annotation_function(enc_output)  # [batch_size, seq_len, encoder_size]
        V = self.value_annotation_function(enc_output)  # [batch_size, seq_len, encoder_size]
        Q = self.query_annotation_function(state_h)  # [batch_size, 1, encoder_size]

        # calculate energy
        energy = self._energy(K, Q)  # [batch_size, seq_len, 1]

        # mask with -inf paddings
        if mask is not None:
            energy.masked_fill_(mask.unsqueeze(-1) == 0, -np.inf)

        # transform energy into probability distribution using softmax
        attention_weights = torch.softmax(energy, dim=1)  # [batch_size, seq_len, 1]

        # for coverage only, calculate the next C
        if type == "coverage":
            self._coverage_compute_next_C(attention_weights, enc_output, state_h)

            # calculate weighted values z (element wise multiplication of energy * values)
        # attention_weights is [batch_size, seq_len, 1], V is [batch_size, seq_len, encoder_size], z is same as V
        z = attention_weights * V  # same as torch.mul(), element wise multiplication

        # finally, calculate context as the esum of z.
        # z is [batch_size, seq_len, encoder_size], context will be [batch_size, encoder_size]
        context = torch.sum(z, dim=1)

        return context, attention_weights  # [batch_size, encoder_size], [batch_size, seq_len, 1]


if __name__ == "__main__":
    print("Testing tokenizer:")

    train_file = "../ro-sts/sts-train.csv"
    dev_file = "../ro-sts/sts-dev.csv"
    test_file = "../ro-sts/sts-test.csv"
    vocab_size = 10000
    train_dev_test_files= [train_file, dev_file, test_file]

    tokenizer = get_tokenizer(train_dev_test_files, vocab_size)

    sentence = "Ion merge prin pădure și culege fragi și mure."
    tokenized_sentence = tokenizer.encode(sentence)

    print("Sentence: {}".format(sentence))
    print("IDs: {}".format(tokenized_sentence.ids))
    print("Tokens: {}".format(tokenized_sentence.tokens))

    # test decoding
    from tokenizers import decoders

    tokenizer.decoder = decoders.WordPiece()
    print("Decoded sentence: {}".format(tokenizer.decode(tokenized_sentence.ids)))