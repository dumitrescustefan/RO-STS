import os, sys, json, copy

import pandas as pd
import numpy as np
"""
1. get a ro-only tsv correspondence file : sim s1_ro s2_ro
2. get a translation dataset: train.ro train.en dev.. test.. with lines
"""

sentences = {}

def get (filename):
    sentences = {}
    dfs=pd.read_excel(filename,engine='openpyxl')
    print(dfs.head())
    for i in range(dfs.shape[0]):
        s1 = "" if dfs.loc[i,'Propozitie originala'] == float("nan") else dfs.loc[i,'Propozitie originala']
        s2 = "" if dfs.loc[i,'Traducere'] == float("nan") else dfs.loc[i,'Traducere']
        sentences[str(dfs.loc[i,'#'])] = ([s1,s2])
    return sentences

sentences.update(get("final/RO-STS_0-2000.xlsx"))
sentences.update(get("final/RO-STS_2000-4000.xlsx"))
sentences.update(get("final/RO-STS_4000-6000.xlsx"))
sentences.update(get("final/RO-STS_6000-8000.xlsx"))
sentences.update(get("final/RO-STS_8000-10000.xlsx"))
sentences.update(get("final/RO-STS_10000-12000.xlsx"))
sentences.update(get("final/RO-STS_12000-14000.xlsx"))
sentences.update(get("final/RO-STS_14000-16000.xlsx"))
sentences.update(get("final/RO-STS_16000-17256.xlsx"))

#print(sentences)
with open("sentence2id.json", "r", encoding="utf8") as f:
    sentence2id = json.load(f)

with open("mapping.json", "r", encoding="utf8") as f:
    mapping = json.load(f)
"""cnt = 0
for (orig, trans) in sentences:
    if orig not in sentence2id:
        print("{}: {}".format(cnt,orig))
    cnt += 1
print(len(sentences))
"""

train = []
dev = []
test = []
train_ro = []
train_en = []
dev_ro = []
dev_en = []
test_ro = []
test_en = []
for k in mapping:
    m=mapping[k]
    index1 = str(m[0])
    index2 = str(m[1])
    if index1 not in sentences:
        print("Index {} not found, skipping pair..".format(index1))
        continue
    if index2 not in sentences:
        print("Index {} not found, skipping pair..".format(index2))
        continue

    s1_en = sentences[str(index1)][0]
    s1_ro = sentences[str(index1)][1]
    s2_en = sentences[str(index2)][0]
    s2_ro = sentences[str(index2)][1]

    if isinstance(s1_en, float) or isinstance(s1_ro, float):
        print("Sentence {} not filled in, skipping pair ..".format(index1))
        continue
    if isinstance(s2_en, float) or isinstance(s2_ro, float):
        print("Sentence {} not filled in, skipping pair ..".format(index2))
        continue

    sim = m[2]
    line = "{}\t{}\t{}\n".format(sim, s1_ro, s2_ro)
    if m[3] == "train":
        train.append(line)
        train_en.append(s1_en + "\n")
        train_en.append(s2_en + "\n")
        train_ro.append(s1_ro + "\n")
        train_ro.append(s2_ro + "\n")
    elif m[3] == "dev":
        dev.append(line)
        dev_en.append(s1_en + "\n")
        dev_en.append(s2_en + "\n")
        dev_ro.append(s1_ro + "\n")
        dev_ro.append(s2_ro + "\n")
    else:
        test.append(line)
        test_en.append(s1_en + "\n")
        test_en.append(s2_en + "\n")
        test_ro.append(s1_ro + "\n")
        test_ro.append(s2_ro + "\n")

with open("RO-STS.train.tsv", "w", encoding="utf8") as f:
    f.writelines(train)
with open("RO-STS.dev.tsv", "w", encoding="utf8") as f:
    f.writelines(dev)
with open("RO-STS.test.tsv", "w", encoding="utf8") as f:
    f.writelines(test)

with open("RO-STS.train.ro", "w", encoding="utf8") as f:
    f.writelines(train_ro)
with open("RO-STS.train.en", "w", encoding="utf8") as f:
    f.writelines(train_en)
with open("RO-STS.dev.ro", "w", encoding="utf8") as f:
    f.writelines(dev_ro)
with open("RO-STS.dev.en", "w", encoding="utf8") as f:
    f.writelines(dev_en)
with open("RO-STS.test.ro", "w", encoding="utf8") as f:
    f.writelines(test_ro)
with open("RO-STS.test.en", "w", encoding="utf8") as f:
    f.writelines(test_en)