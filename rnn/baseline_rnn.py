import os, sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from utils_rnn import get_tokenizer, get_dataloaders
from model_rnn import STSRNNBaselineModel

# global fixed values
train_file = "../ro-sts/sts-train.csv"
dev_file = "../ro-sts/sts-dev.csv"
test_file = "../ro-sts/sts-test.csv"
vocab_size = 10000
batch_size = 16

# load or create tokenizer
tokenizer = get_tokenizer([train_file, dev_file, test_file], vocab_size)

# load data
train_dataloader, val_dataloader, test_dataloader = get_dataloaders(train_file, dev_file, test_file, tokenizer, batch_size)
#print("Train set has {} instances, meaning {:.0f} steps.".format(len(train_dataset), len(train_dataset) / batch_size))
#print("Valid set has {} instances, meaning {:.0f} steps.".format(len(val_dataset), len(val_dataset) / batch_size))
#print("Test set has  {} instances, meaning {:.0f} steps.".format(len(test_dataset), len(test_dataset) / batch_size))

# instantiate model
model = STSRNNBaselineModel(vocab_size)


early_stop = EarlyStopping(
    monitor='valid/pearson',
    patience=3,
    strict=True,
    verbose=True,
    mode='max'
)

trainer = pl.Trainer(
    gpus=0,
    #checkpoint_callback=checkpoint_callback,
    callbacks=[early_stop],
    limit_train_batches=20,
    limit_val_batches=10,
    accumulate_grad_batches=8,
    weights_save_path='model',
    gradient_clip_val=1.0,
    #logger=wandb_logger,
    auto_lr_find=False,
    # val_check_interval=0.25,
    # progress_bar_refresh_rate=10,
)

trainer.fit(model, train_dataloader, val_dataloader)

# trainer.test(model, test_dataloader)