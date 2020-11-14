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