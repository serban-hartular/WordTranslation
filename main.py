import math
from typing import Callable

import numpy as np
from numpy import dot
from numpy.linalg import norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import random

from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel

import utils


def noise(arr, k=0.5):
    for r in arr:
        for i in range(len(r)):
            r[i] += (random.random() - 0.5)*k

import pandas as pd
from bert_embeddings import bert_encode_text

def find_best_match(target_vec : np.ndarray, options : list[tuple[str, str, np.ndarray]],
                    metricFn : Callable[[np.ndarray, np.ndarray], float]=None)\
        -> list[tuple[str, str, float]]:
    if metricFn is None:
        metricFn = lambda a, b : math.sqrt(dot(a-b, a-b))
    results = [(word, uid, metricFn(target_vec, _vec)) for word, uid, _vec in options]
    results.sort(key=lambda t: t[-1]) # sort by score
    return results

def make_predictor(PredictorClass, X, Y) -> (object, float):
    pred = PredictorClass()
    pred.fit(X, Y)
    Ypred = pred.predict(X)
    return pred, r2_score(Y, Ypred)

if __name__ == '__main__':
    load_local = True
    if load_local:
        print("Loading bert resources (local)")
        en_tokenizer : BertTokenizer = utils.p_load('bert_lang_model/bert-base-english-cased-tokenizer.p')
        en_model : BertModel = utils.p_load('bert_lang_model/bert-base-english-cased-model.p')
        ro_tokenizer : BertTokenizer = utils.p_load('bert_lang_model/bert-base-romanian-cased-v1-tokenizer.p')
        ro_model : BertModel = utils.p_load('bert_lang_model/bert-base-romanian-cased-v1-model.p')
    else:
        print('Loading english')
        en_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        en_model = BertModel.from_pretrained('bert-base-cased')
        print('Loading romanian')
        ro_tokenizer = BertTokenizer.from_pretrained('dumitrescustefan/bert-base-romanian-cased-v1')
        ro_model = BertModel.from_pretrained('dumitrescustefan/bert-base-romanian-cased-v1')
    print('Loading words')
    word_pair_data = utils.p_load('trash/en_ro_vecs202_singleword.p')
    X = word_pair_data['en']
    Y = word_pair_data['ro']
    lg = LinearRegression()
    lg.fit(X, Y)
    Ypred = lg.predict(X)
    print("R2 = ", r2_score(Y, Ypred))
    texts = ('The hallway smelt of boiled cabbage and old rag mats .',
                        'Holul blocului mirosea a varză călită și a preșuri vechi .')
    en_text, ro_text = texts
    word_vecs = {}
    models = {'en':{'tokenizer':en_tokenizer, 'model':en_model},
              'ro':{'tokenizer':ro_tokenizer, 'model':ro_model}}
    for lang, text in zip(('en', 'ro'), texts):
        word_vecs[lang] = []
        tokens = text.split(' ')
        for token in tokens:
            vector = bert_encode_text(token, models[lang]['model'], models[lang]['tokenizer'])
            vector = vector[0][1] # only item of list, 2nd item of tuple
            word_vecs[lang].append((token, vector))

    matches = {}
    for en_word, en_vec in word_vecs['en']:
        ro_vec_predicted = lg.predict([en_vec])[0]
        matches[en_word] = find_best_match(ro_vec_predicted, word_vecs['ro'],
                                lambda a, b : 1 - dot(a, b)/(norm(a)*norm(b)))
    for en_word, match in matches.items():
        match_str = '\t'.join([f'{w}: {score:.2f}' for w, score in match])
        print(f'{en_word}\t{match[0][0]}\t{match_str}')
    # words = pd.read_csv('./en_ro_singleword_500.csv', encoding='utf-8', sep='\t')
    # rows = words.to_dict(orient="records")
    # X = []
    # Y = []
    # langs = ['en', 'ro']
    # tokenizers = [en_tokenizer, ro_tokenizer]
    # models = [en_model, ro_model]
    # matrices = [X, Y]
    # for i, row in enumerate(rows):
    #     print(f'Row {i+1} of {len(rows)}')
    #     for lang, tok, model, matrix in zip(langs, tokenizers, models, matrices):
    #         word = row[lang]
    #         token, word_vec = encode_text(word, tok, model)[0]
    #         if token != word:
    #             print(f'Error, returned token {token} different from word {word}')
    #         matrix.append(np.array(word_vec))
    # print('To numpy arrays')
    # X = np.array(X)
    # Y = np.array(Y)
