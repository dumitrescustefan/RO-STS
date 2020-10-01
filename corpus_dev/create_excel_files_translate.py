import os, sys, json, copy
import xlsxwriter

def process_lines(lines, sentence2id, mapping, type):
    for line in lines:
        # main-captions	MSRvid	2012test	0024	2.500	A girl is styling her hair.	A girl is brushing her hair.
        parts = line.strip().split("\t")
        #print(parts[0:4])
        s1 = parts[5]
        s2 = parts[6]
        sim = float(parts[4])
        meta = "^".join(parts[0:4])

        sentence2id.append(s1)
        sentence2id.append(s2)
        mapping[meta] = [len(sentence2id)-2, len(sentence2id)-1, sim, type]

    return sentence2id, mapping

def write_excel(start, end, sentence2id, id2sentence, mapping):
    def copy_format(book, fmt):
        properties = [f[4:] for f in dir(fmt) if f[0:4] == 'set_']
        dft_fmt = book.add_format()
        return book.add_format({k: v for k, v in fmt.__dict__.items() if k in properties and dft_fmt.__dict__[k] != v})

    filename = "RO-STS_{}-{}-translated.xlsx".format(start, end)
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()

    header_row_format = workbook.add_format()
    header_row_format.set_bottom(1)
    header_row_format.set_bottom_color('#666666')
    header_row_format.set_bg_color('#CCCCCC')
    header_row_format.set_bold(True)

    # row formats
    row_format = workbook.add_format()
    row_format.set_bottom(1)
    row_format.set_bottom_color('#aaaaaa')
    row_format.set_bg_color('#ffffff')
    row_format.set_align('vcenter')

    alt_row_format = copy_format(workbook, row_format)
    alt_row_format.set_bg_color('#f0faff')

    # cell formats
    cell_format = workbook.add_format()
    cell_format.set_right(1)
    cell_format.set_right_color('#dddddd')
    cell_format.set_bottom(1)
    cell_format.set_bottom_color('#aaaaaa')
    cell_format.set_align('vcenter')

    alt_cell_format = copy_format(workbook, cell_format)
    alt_cell_format.set_bg_color('#f0faff')

    en_cell_format = copy_format(workbook, cell_format)
    en_cell_format.set_shrink()

    en_alt_cell_format = copy_format(workbook, alt_cell_format)
    en_alt_cell_format.set_shrink()

    cell_format.set_right(0)
    alt_cell_format.set_right(0)

    worksheet.set_default_row(25)
    worksheet.set_column('A:A', 6)
    worksheet.set_column('B:B', 2)
    worksheet.set_column('C:C', 80)
    worksheet.set_column('D:D', 50)
    worksheet.set_column('E:E', 70)
    
    worksheet.write(0, 0, "#")
    worksheet.write(0, 2, "Propozitie originala")
    worksheet.write(0, 3, "Traducere")
    worksheet.write(0, 4, "Google-translate")
    
    worksheet.set_row(0, None, cell_format = header_row_format)

    row = 1
    for index in range(start, end):
        
        worksheet.write(row, 0, index)
        if index % 2 == 0:
            worksheet.set_row(row, None, cell_format=alt_row_format)
            worksheet.write(row, 2, sentence2id[index], en_alt_cell_format)
            worksheet.write(row, 3, "", alt_cell_format)
            worksheet.write(row, 4, trans[sentence2id[index]], alt_cell_format)
        else:
            worksheet.set_row(row, None, cell_format=row_format)
            worksheet.write(row, 2, sentence2id[index], en_cell_format)
            worksheet.write(row, 3, "", cell_format)
            worksheet.write(row, 4, trans[sentence2id[index]], en_cell_format)


        row += 1

    workbook.close()


sentence2id = []
mapping = {}

with open("stsbenchmark/sts-train.csv","r", encoding="utf8") as f:
    sentence2id, mapping = process_lines(f.readlines(), sentence2id, mapping, "train")
    print("Unique sents: {}, mapping len: {}".format(len(sentence2id), len(mapping)))

with open("stsbenchmark/sts-dev.csv","r", encoding="utf8") as f:
    sentence2id, mapping = process_lines(f.readlines(), sentence2id, mapping, "dev")
    print("Unique sents: {}, mapping len: {}".format(len(sentence2id), len(mapping)))

with open("stsbenchmark/sts-test.csv","r", encoding="utf8") as f:
    sentence2id, mapping = process_lines(f.readlines(), sentence2id, mapping, "test")
    print("Unique sents: {}, mapping len: {}".format(len(sentence2id), len(mapping)))

with open("sentence2id.json", "w", encoding="utf8") as f:
    json.dump(sentence2id, f, indent=2)

with open("mapping.json", "w", encoding="utf8") as f:
    json.dump(mapping, f, indent=2)


id2sentence = []
for sentence in sentence2id:
    id2sentence.append(sentence[0])

print("Unique sents: {}, total sentences: {}".format(len(sentence2id), len(mapping)))

"""
from googletrans import Translator
translator = Translator()
translations = translator.translate(sentence2id, dest='ro')
trans = {}
for translation in translations:
    trans[translation.origin] = translation.text
print("trans done")
import json
with open("trans.json", "w", encoding="utf8") as f:
    json.dump(trans, f)

print("done")
"""
import json
import time
"""
with open("trans.json", "r", encoding="utf8") as f:
    trans = json.load(f)
    
from googletrans import Translator
translator = Translator()

while True:
    english = []
    for k in trans:
        if k == trans[k]:
            english.append(k)
    if len(english) == 0:
        break
        
    print("batch translate {}".format(len(english)))
    for i in range(0,len(english), 100):
        print(str(i)+"...")
        translations = translator.translate(english[i:min(i+100, len(english))], src = 'en', dest='ro')
       
        for translation in translations:
            trans[translation.origin] = translation.text
            print(str(i)+"\t"+translation.text+"\t\t"+translation.origin)

        time.sleep(1)


    with open("trans.json", "w", encoding="utf8") as f:
        json.dump(trans, f)

print("done")
"""

with open("trans.json", "r", encoding="utf8") as f:
    trans = json.load(f)

for i in range(0, 16000, 2000):
    write_excel(i,i+2000, sentence2id, id2sentence, mapping)

write_excel(16000,len(sentence2id), sentence2id, id2sentence, mapping) # last chunk
