"""
Three metrics are used to evaluate the model: ROUGE, BLEU and cosine similarity.

This script generates two files with the results of the evaluation of the model.

- "modelname".csv: line by line results of the evaluation of the model.
- "modelname".txt: overall results of the evaluation of the model.

Two input files are needed to run this script:

- model_name: name of the model to evaluate.
- input_file: path to the test dataset.

To run it, type in the terminal: python metrics.py

"""


import json
import evaluate
import statistics
import csv

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util


model_name = 'TF-S_go_2023-06-03'
input_file = '../datasets/go/test.json'


path = f'../models/{model_name}'  # change if needed
rouge = evaluate.load('rouge')
bleu = evaluate.load('bleu')
sentencetrans = SentenceTransformer('all-MiniLM-L6-v2')


def getData():
    with open(input_file, encoding='utf8') as f:
        for row in f:
            yield (json.loads(row)['code'], json.loads(row)['nl'])



def main():
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path)
    hypotheses = []
    references = []


    for code, nl in getData():
        tokens = tokenizer(code, return_tensors='pt', max_length=512)
        output = model.generate(**tokens)
        prediction = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

        hypotheses.append(prediction.replace('\n', '').replace('\r', '').lower())
        references.append(nl.replace('\n', '').replace('\r', '').lower())


    #Compute embedding for both lists
    embeddings1 = sentencetrans.encode(references, convert_to_tensor=True)
    embeddings2 = sentencetrans.encode(hypotheses, convert_to_tensor=True)

    #Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2)

    with open(f"{model_name}.csv", "a", newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["REFERENCE", "MODEL", "ROUGE1", "ROUGE2", "ROUGEL", "BLEU1", "BLEU2", "BLEU3", "BLEU4", "COS"])
        writer.writeheader()

        cosines = []
        for i in range(len(hypotheses)):
            hyp = [hypotheses[i]]
            ref = [references[i]]

            rouge_results = rouge.compute(predictions=hyp, references=ref)
            bleu1_results = bleu.compute(predictions=hyp, references=ref, max_order=1)
            bleu2_results = bleu.compute(predictions=hyp, references=ref, max_order=2)
            bleu3_results = bleu.compute(predictions=hyp, references=ref, max_order=3)
            bleu4_results = bleu.compute(predictions=hyp, references=ref, max_order=4)

            cosin = round(100*float(cosine_scores[i][i]), 2)
            if cosin < 0:
                cosin = 0

            cosines.append(cosin)
            writer.writerow({"REFERENCE": references[i],
                            "MODEL": hypotheses[i], 
                            "ROUGE1": round(100*rouge_results["rouge1"], 2),
                            "ROUGE2": round(100*rouge_results["rouge2"], 2),
                            "ROUGEL": round(100*rouge_results["rougeL"], 2),
                            "BLEU1":  round(100*bleu1_results["bleu"], 2),
                            "BLEU2":  round(100*bleu2_results["bleu"], 2),
                            "BLEU3":  round(100*bleu3_results["bleu"], 2),
                            "BLEU4":  round(100*bleu4_results["bleu"], 2),
                            "COS":    cosin})


    rouge_results = rouge.compute(predictions=hypotheses, references=references)
    bleu1_results = bleu.compute(predictions=hypotheses, references=references, max_order=1)
    bleu2_results = bleu.compute(predictions=hypotheses, references=references, max_order=2)
    bleu3_results = bleu.compute(predictions=hypotheses, references=references, max_order=3)
    bleu4_results = bleu.compute(predictions=hypotheses, references=references, max_order=4)

    with open(f'{model_name}.txt', 'w', encoding='utf-8') as fp:
        fp.write(f'Rouge results:\n{rouge_results}\n')
        fp.write(f'Bleu results:\n{bleu1_results}\n{bleu2_results}\n{bleu3_results}\n{bleu4_results}\n')
        fp.write(f'Cosine quartiles: {[round(q, 1) for q in statistics.quantiles(cosines, n=4)]}')


if __name__ == '__main__':
    main()
