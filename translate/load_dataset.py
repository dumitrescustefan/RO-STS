from datasets import list_datasets, load_dataset, list_metrics, load_metric
import os


def save_dataset(outpath_en, outpath_ro, subset):
    with open(outpath_en, "w", encoding="utf-8") as outfile_en, open(outpath_ro, "w", encoding="utf-8") as outfile_ro:
        for translations in wmt_dataset[subset]["translation"]:
            outfile_en.write(translations["en"] + "\n")
            outfile_ro.write(translations["ro"] + "\n")


# Load a dataset and print the first examples in the training set
wmt_dataset = load_dataset('wmt16', 'ro-en')

if not os.path.exists("data"):
    os.makedirs("data")

save_dataset("data/train-en.txt", "data/train-ro.txt", "train")
save_dataset("data/valid-en.txt", "data/valid-ro.txt", "validation")
save_dataset("data/test-en.txt", "data/test-ro.txt", "test")
