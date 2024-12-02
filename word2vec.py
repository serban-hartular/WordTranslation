import dataclasses
import math
from typing import Callable

import gensim
from gensim.models.keyedvectors import KeyedVectors
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import main
import utils
import numpy as np


def find_best_match(target_vec : np.ndarray, options : list[dict],
                    metricFn : Callable[[np.ndarray, np.ndarray], float]=None)\
        -> list[dict]:
    if metricFn is None:
        metricFn = lambda a, b : math.sqrt(dot(a-b, a-b))
    def score(d):
        if not any(d['vector']):
            d['score'] = 100000000
        else:
            d['score'] = metricFn(target_vec, d['vector'])
    list(map(score, options))
    results = list(options)
    results.sort(key=lambda d: d['score']) # sort by score
    return results

def get_vector(word : str, model : KeyedVectors) -> np.ndarray|None:
    if word not in model:
        return None
    return model[word]

def get_text_vector(text : str|list[str], model : KeyedVectors) -> list[tuple[str, np.ndarray]]:
    if isinstance(text, str):
        text = text.split(' ')
    return [(word, get_vector(word, model)) for word in text]


if __name__ == '__main__':
    print('Loading vectors')
    model_ro = utils.p_load('word2vec_lang_model/corola.300.20.p')
    # model_en = KeyedVectors.load_word2vec_format('./word2vec_lang_model/GoogleNews-vectors-negative300.bin', binary=True)
    model_en = utils.p_load('word2vec_lang_model/google_word2vec_big.p')
    print('Loading words')
    words = pd.read_csv('word_lists/en_ro_singleword_500.csv', encoding='utf-8', sep='\t')
    rows = words.to_dict(orient="records")
    X = []
    Y = []
    langs = ['en', 'ro']
    models = [model_en, model_ro]
    matrices = [X, Y]
    print('Building training matrices.')
    for i, row in enumerate(rows):
        print(f'Row {i+1} of {len(rows)}')
        vectors = [None, None]
        for i, (lang, model) in enumerate(zip(langs, models)):
            word = row[lang]
            if word not in model:
                print(f'{word} not found.')
                continue
            vectors[i] = model[word]
        if all([v is not None for v in vectors]):
            for i, matrix in enumerate(matrices):
                matrix.append(np.array(vectors[i]))
    print('Done')
    #
    # lg = LinearRegression()
    # lg.fit(X, Y)
    # Ypred = lg.predict(X)
    # print("R2 = ", r2_score(Y, Ypred))
    # lg : LinearRegression = utils.p_load('./en_ro450_word2vec_linear_regressor.p')
    # models = [model_en, model_ro]
    #
    # chunk_tokens_lang = utils.p_load('./en_ro_1984_paired_tokens.p')
    #
    # chunk_tokens_with_options = []
    #
    # vector_len = None
    #
    # for chunk_pair in chunk_tokens_lang:
    #     # doing one paired chunk
    #     en_text = ' '.join([n['form'] for n in chunk_pair['en']])
    #     ro_text = ' '.join([n['form'] for n in chunk_pair['ro']])
    #     nouns = {}
    #     for lang, model in zip(('en', 'ro'), models):
    #         nouns[lang] = [n for n in chunk_pair[lang] if n['upos'] == 'NOUN']
    #         for token in nouns[lang]:
    #             key = token['form'] if lang == 'en' else token['form'].lower()
    #             if key in model:
    #                 token['vector'] = model[key]
    #                 vector_len = len(token['vector'])
    #             else:
    #                 token['vector'] = [0.0]*vector_len
    #     matches = {}
    #     for en_word in nouns['en']:
    #         ro_vec_predicted = lg.predict([en_word['vector']])[0]
    #         options = find_best_match(ro_vec_predicted, nouns['ro'],
    #                                                 lambda a, b: 1 - dot(a, b) / (norm(a) * norm(b)))
    #         matches[en_word['chunk_id']] = options
    #     # if we have more english nouns than romanian nouns, some english nouns have no match
    #     if len(nouns['en']) > len(nouns['ro']) and len(nouns['ro']) > 1:
    #         len_diff = len(nouns['en']) > len(nouns['ro'])
    #         ordered_noun_matches = list(matches.items())
    #         ordered_noun_matches.sort(key=lambda t : t[1][0]['score'])
    #         #that is: given a (en_word, matching_ro_list) tuple, select the first (with the highest score) and return its score
    #         for word, match_list in ordered_noun_matches[:len_diff]:
    #             match_list.insert(0, ({'form':'', 'chunk_id':-1, 'score':-1}))
    #     # now remove scores from the list, keeping only the words
    #     matches = {_w : [d['chunk_id'] for d in _mlist] for _w, _mlist in matches.items()}
    #     # now add to the final list
    #     chunk_tokens_with_options.append({
    #         'en' : en_text,
    #         'ro' : ro_text,
    #         'options' : matches
    #     })
    #
    #
