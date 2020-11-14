from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize
from rouge.rouge import Rouge
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("gold_file")
    parser.add_argument("pred_file")

    args = parser.parse_args()

    list_references = []
    with open(args.gold_file, "r", encoding="utf-8") as file:
        for line in file:
            list_references.append(line[:-1])

    list_hypotheses = []
    with open(args.pred_file, "r", encoding="utf-8") as file:
        for line in file:
            list_hypotheses.append(line[:-1])

    print("Bleu score: {:.2f}\n".format(corpus_bleu([[word_tokenize(ref)] for ref in list_references],
                                                    [word_tokenize(hyp) for hyp in list_hypotheses])
                                                    * 100))

    rouge = Rouge()
    hyps, refs = map(list, zip(*[[hyp, ref] for hyp, ref in zip(list_hypotheses, list_references)]))
    rogue_scores = rouge.get_scores(hyps, refs, avg=True)

    print("Rogue-1:\n\tP: {:.2f}\n\tR: {:.2f}\n\tF: {:.2f}\n".format(rogue_scores["rouge-1"]["p"] * 100,
                                                                     rogue_scores["rouge-1"]["r"] * 100,
                                                                     rogue_scores["rouge-1"]["f"] * 100))

    print("Rogue-2:\n\tP: {:.2f}\n\tR: {:.2f}\n\tF: {:.2f}\n".format(rogue_scores["rouge-2"]["p"] * 100,
                                                                     rogue_scores["rouge-2"]["r"] * 100,
                                                                     rogue_scores["rouge-2"]["f"] * 100))

    print("Rogue-l:\n\tP: {:.2f}\n\tR: {:.2f}\n\tF: {:.2f}\n".format(rogue_scores["rouge-l"]["p"] * 100,
                                                                     rogue_scores["rouge-l"]["r"] * 100,
                                                                     rogue_scores["rouge-l"]["f"] * 100))