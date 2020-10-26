from datasets import list_datasets, load_dataset, list_metrics, load_metric
import os


def save_dataset(outpath_en, outpath_ro, subset):
    with open(outpath_en, "w", encoding="utf-8") as outfile_en, open(outpath_ro, "w", encoding="utf-8") as outfile_ro:
        for translations in wmt16_roen[subset]["translation"]:
            outfile_en.write(translations["en"] + "\n")
            outfile_ro.write(translations["ro"] + "\n")


wmt16_roen = load_dataset('wmt16', "ro-en")

if not os.path.exists("data"):
    os.makedirs("data")

save_dataset("data/train_en.txt", "data/train_ro.txt", "train")
save_dataset("data/valid_en.txt", "data/valid_ro.txt", "validation")
save_dataset("data/test_en.txt", "data/test_ro.txt", "test")