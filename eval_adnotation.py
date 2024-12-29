from __future__ import annotations

import dataclasses
import warnings
from copy import deepcopy
from typing import Callable, Any, List
import pandas as pd

import numpy as np
import sklearn.base
from numpy.linalg import norm

# import bert_embeddings
import utils
# import word2vec

ANNOTATIONS_KEY = 'annotations'

Token = dict

@dataclasses.dataclass
class TextChunk:
    token_list : list[Token]
    candidates : list[int]
    vectors : list[np.ndarray] = None
    meta : dict[str, str] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
        # return {'token_list':self.token_list,
        #         'candidates':self.candidates,
        #         'vectors':self.vectors,
        #         'meta':self.meta}
    @staticmethod
    def from_dict(d : dict) -> 'TextChunk':
        return TextChunk(**d)
    def __str__(self):
        return ' '.join([str(tok.get('form')) for tok in self.token_list])
    def __repr__(self):
        return repr(str(self))
    def get_vector(self, tok_index : int) -> np.ndarray|None:
        index_in_candidates = self.candidates.index(tok_index)
        if index_in_candidates < 0:
            return None
        return self.vectors[index_in_candidates]

    def get_word(self, tok_index : int) -> str:
        return self.token_list[tok_index]['form']

    def get_word_list(self) -> list[str]:
        return [tok['form'] for tok in self.token_list]

@dataclasses.dataclass
class TextPair:
    source : TextChunk
    target : TextChunk
    linked_tok_ids : dict[int, int|str]
    meta : dict[str, str] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict:
        return {'source':self.source.to_dict(),
                'target':self.target.to_dict(),
                'linked_tok_ids':self.linked_tok_ids,
                'meta':self.meta}
    @staticmethod
    def from_dict(d : dict) -> 'TextPair':
        return TextPair(source=TextChunk.from_dict(d['source']),
                        target=TextChunk.from_dict(d['target']),
                        linked_tok_ids=d['linked_tok_ids'],
                        meta=d['meta']
                        )


    def copy(self) -> TextPair:
        return TextPair.from_dict(deepcopy(self.to_dict()))

    def __str__(self):
        return '\n'.join([str(self.source), str(self.target), str(self.linked_tok_ids)])
    def __repr__(self):
        return repr(str(self))

class ParallelTexts(List[TextPair]):
    def __init__(self, tp_list : list[TextPair]=None, meta : dict = None):
        super().__init__(tp_list if tp_list else [])
        self.meta = meta if meta else {}

    def to_dict(self) -> dict:
        return {'text_pairs':[tp.to_dict() for tp in self], 'meta':self.meta}

    @staticmethod
    def from_dict(d : dict) -> 'ParallelTexts':
        if 'text_pairs' not in d:
            raise Exception('Error! Dict does not contain text pairs!')
        text_pairs = [TextPair.from_dict(tp) for tp in d['text_pairs']]
        return ParallelTexts(text_pairs, d.get('meta'))

    def copy(self) -> 'ParallelTexts':
        return ParallelTexts.from_dict(deepcopy(self.to_dict()))

    def pickle(self, filename : str):
        utils.p_save(self.to_dict(), filename)

    @staticmethod
    def from_pickle(filename : str) -> 'ParallelTexts':
        d = utils.p_load(filename)
        return ParallelTexts.from_dict(d)


def guess_closest_match(estimate : np.ndarray, targets : list[tuple[int, np.ndarray]],
                        metric : Callable[[np.ndarray, np.ndarray], float] = None)\
        -> list[tuple[int, float]]:
    if metric is None:
        metric = lambda v1, v2 : np.linalg.norm(v2 - v1)
    result = [(id, metric(estimate, vector)) for id, vector in targets]
    result.sort(key=lambda t : t[1]) # t[1] is the score given by the metric
    return result

def text_pair_guess_annotations(tp : TextPair, translator : sklearn.base.BaseEstimator,
                                threshold : float = None,
                                 metric : Callable[[np.ndarray, np.ndarray], float] = None)\
        -> dict[int, list[int]]:
    if not tp.source.candidates or not tp.target.candidates:
        return {}
    translated_vectors = translator.predict(tp.source.vectors)
    guess_dict = {}
    guess_candidates = list(zip(tp.target.candidates, tp.target.vectors))
    for source_id, translated_vec in zip(tp.source.candidates, translated_vectors):
        result = guess_closest_match(translated_vec, guess_candidates, metric)
        guess_dict[source_id] = [t[0] for t in result]
        # insert 'none_guess' if top distance is greater than threshold
        if not result or (threshold is not None and result[0][1] > threshold):
            guess_dict[source_id].insert(0, 'none_guess')
    return guess_dict

def eval_text_pair_guess_scores(tp : TextPair, translator : sklearn.base.BaseEstimator,
                                 metric : Callable[[np.ndarray, np.ndarray], float] = None)\
        -> dict[str, list[float]]:
    if tp.meta.get(ANNOTATIONS_KEY) != 'manual':
        return {}
    metric_dict = {'right':[], 'wrong':[], 'none':[]}
    if not tp.source.candidates or not tp.target.candidates:
        return metric_dict
    translated_vectors = translator.predict(tp.source.vectors)
    guess_candidates = list(zip(tp.target.candidates, tp.target.vectors))
    for source_id, translated_vec in zip(tp.source.candidates, translated_vectors):
        if source_id not in tp.linked_tok_ids:
            print('Error! source candidate not in annotation!')
            continue
        correct_answer = tp.linked_tok_ids[source_id]
        result = guess_closest_match(translated_vec, guess_candidates, metric)
        best_guess_score = result[0]
        # now, is the guess right, wrong, or is the answer 'none'
        if correct_answer == best_guess_score[0]:
            metric_dict['right'].append(best_guess_score[1])
        elif not isinstance(correct_answer, int) or correct_answer < 0:
            metric_dict['none'].append(best_guess_score[1])
        else: # just wrong
            metric_dict['wrong'].append(best_guess_score[1])
    return metric_dict

def eval_parallel_text_guess_scores(tp_list : list[TextPair],
                                    translator : sklearn.base.BaseEstimator,
                                    metric : Callable[[np.ndarray, np.ndarray], float] = None) \
        -> dict[str, list[float]]:
    metric_dict = {'right':[], 'wrong':[], 'none':[]}
    for tp in tp_list:
        if tp.meta.get(ANNOTATIONS_KEY) != 'manual':
            continue
        tp_dict = eval_text_pair_guess_scores(tp, translator, metric)
        for result_type in metric_dict:
            metric_dict[result_type].extend(tp_dict[result_type])
    return metric_dict


def evaluate_estimator(text_pairs : list[TextPair], translator : sklearn.base.BaseEstimator,
                                 metric : Callable[[np.ndarray, np.ndarray], float] = None) -> (int, int):

    count, score = 0, 0
    for tp in text_pairs:
        if not tp.linked_tok_ids:
            continue
        guesses = text_pair_guess_annotations(tp, translator, metric)
        for c_id, correct_eq in tp.linked_tok_ids.items():
            if not isinstance(correct_eq, int) or correct_eq < 0:
                continue
            count += 1
            # compare with best guess
            best_guess = guesses[c_id][0]
            if best_guess == correct_eq or\
                    tp.target.token_list[best_guess]['form'] == tp.target.token_list[correct_eq]['form']:
                score += 1
    return score, count


def evaluate_adaptive_estimator(text_pairs : list[TextPair], TranslatorClass,
                                init_X : list[np.ndarray] = None, init_Y : list[np.ndarray] = None,
                                 metric : Callable[[np.ndarray, np.ndarray], float] = None)\
        -> list[int]:
    result_list : list[int] = []
    X = [] if not init_X else list(init_X)
    Y = [] if not init_Y else list(init_Y)
    for tp in text_pairs:
        if not tp.linked_tok_ids:
            continue
        # guesses = text_pair_guess_annotations(tp, translator, metric)
        for i, (c_id, correct_eq) in enumerate(tp.linked_tok_ids.items()):
            if not isinstance(correct_eq, int) or correct_eq < 0:
                continue
            if not(X and Y):
                result_list.append(0)
            else:
                translator = TranslatorClass()
                translator.fit(X, Y)
                translated_vec = translator.predict([tp.source.vectors[i]])
                candidates = list(zip(tp.target.candidates, tp.target.vectors))
                guesses_ranked = guess_closest_match(translated_vec, candidates, metric)
                best_guess = guesses_ranked[0][0]
                if best_guess == correct_eq or\
                        tp.target.token_list[best_guess]['form'] == tp.target.token_list[correct_eq]['form']:
                    result_list.append(1)
                else:
                    result_list.append(0)
            X.append(tp.source.vectors[i])
            correct_equivalent_index = tp.target.candidates.index(correct_eq)
            Y.append(tp.target.vectors[correct_equivalent_index])
    return result_list

def cosine_difference(a, b) -> float:
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    return 1 - np.dot(a, b) / norm(a) / norm(b)

def get_vectors_from_wordlist(word_list : list[str], get_vec_fn : Callable[[str, ...], float],
                              fn_args : list) -> list[float]:
    return [get_vec_fn(word, *fn_args) for word in word_list]

def get_training_set_from_text_pair(tp : TextPair) -> tuple[list[np.ndarray], list[np.ndarray]]:
    source_list, target_list = [], []
    for src_id, target_id in tp.linked_tok_ids.items():
        if not isinstance(target_id, int) or target_id < 0:
            continue
        source_list.append(tp.source.get_vector(src_id))
        if source_list[-1] is None:
            print(f'Error, could not find vector of token {src_id}, "{tp.source.get_word(src_id)}"')
        target_list.append(tp.target.get_vector(target_id))
        if target_list[-1] is None:
            print(f'Error, could not find vector of token {target_id}, "{tp.target.get_word(target_id)}"')
    return source_list,target_list

def get_training_set_from_parallel_texts(tp_list : list[TextPair],
                                         annotations = 'manual') -> tuple[list[np.ndarray], list[np.ndarray]]:
    source_list, target_list = [], []
    for tp in tp_list:
        if annotations is not None and tp.meta.get(ANNOTATIONS_KEY) != annotations:
            continue
        src_add, target_add = get_training_set_from_text_pair(tp)
        source_list.extend(src_add)
        target_list.extend(target_add)
    return source_list,target_list


def training_set_exclude_nulls(X : list[np.ndarray|None], Y : list[np.ndarray|None])\
        -> (list[np.ndarray], list[np.ndarray]):
    vector_tuples = [(x_vec, y_vec) for x_vec, y_vec in zip(X, Y) if x_vec is not None and y_vec is not None]
    X = [v[0] for v in vector_tuples]
    Y = [v[1] for v in vector_tuples]
    return X, Y


def get_training_set_from_wordlists(wordlist_df : pd.DataFrame, label1 : str, label2 : str,
                        get_vec_fn : Callable[[str, ...], float], fn_args : list)\
        -> (list[np.ndarray], list[np.ndarray]):
    # singleword_pairs = pd.read_csv('word_lists/en_ro_singleword_500.csv', sep='\t', encoding='utf-8')
    df_dict = wordlist_df.to_dict(orient='series')
    src_words, target_words = [df_dict[lang].to_list() for lang in (label1, label2)]
    X_wordlist = get_vectors_from_wordlist(src_words, get_vec_fn, fn_args)
    Y_wordlist = get_vectors_from_wordlist(target_words, get_vec_fn, fn_args)
    X_wordlist, Y_wordlist = training_set_exclude_nulls(X_wordlist, Y_wordlist)
    return X_wordlist, Y_wordlist

def chunk_update_vector_data(chunk : TextChunk,
                       get_list_vec_fn : Callable[[str|list[str], ...], tuple[str, np.ndarray]],
                       fn_args : list):
    words = chunk.get_word_list()
    word_vectors = get_list_vec_fn(words, *fn_args)
    chunk.vectors = [word_vectors[c_id][1] for c_id in chunk.candidates]

if __name__ == "__main__":
    from sklearn.linear_model import LinearRegression
    # print('Load word vec language models')
    # word2vec_ro = utils.p_load('word2vec_lang_model/corola.300.20.p')
    # word2vec_en = utils.p_load('word2vec_lang_model/google_word2vec_big.p')
    # base_bert_en_tokenizer: BertTokenizer = utils.p_load('bert_lang_model/bert-base-english-cased-tokenizer.p')
    # base_bert_en_model: BertModel = utils.p_load('bert_lang_model/bert-base-english-cased-model.p')
    # bert1_ro_tokenizer: BertTokenizer = utils.p_load('bert_lang_model/bert-base-romanian-cased-v1-tokenizer.p')
    # bert1_ro_model: BertModel = utils.p_load('bert_lang_model/bert-base-romanian-cased-v1-model.p')
    # roBERT_ro_tokenizer: BertTokenizer = utils.p_load('bert_lang_model/RoBERT-large-model.p')
    # roBERT_ro_model: BertModel = utils.p_load('bert_lang_model/RoBERT-large-tokenizer.p')

    print('Loading premade word lists for training')
    wordlist_word2vec : dict[str, list[np.ndarray]] = utils.p_load('./vectors_pickled/wordlist_vecs_word2vec.p')
    wordlist_bert1 : dict[str, list[np.ndarray]] = utils.p_load('./vectors_pickled/wordlist_vecs_bert.p')
    wordlist_roBERT: dict[str, list[np.ndarray]] = utils.p_load('./vectors_pickled/wordlist_vecs_roBERT-large.p')

    print('Load parallel text data')
    en2ro1984_word2vec =  ParallelTexts.from_pickle('./1984_pickled/en2ro-1984-word2vec.p')
    en2ro1984_bert1 =  ParallelTexts.from_pickle('./1984_pickled/en2ro-1984-bert1.p')
    en2ro1984_roBERT = ParallelTexts.from_pickle('./1984_pickled/en2ro-1984-roBERT.p')

    combo_dict = {
        'word2vec': {
            # 'en': [word2vec_en], 'ro':[word2vec_ro],
                     'wordlist':wordlist_word2vec, 'text':en2ro1984_word2vec},
        'bert1': {
                    # 'en': [base_bert_en_model, base_bert_en_tokenizer],
                    # 'ro': [bert1_ro_model, bert1_ro_tokenizer],
                  'wordlist':wordlist_bert1, 'text':en2ro1984_bert1},
        'RoBERT':{
                  #   'en': [base_bert_en_model, base_bert_en_tokenizer],
                  # 'ro': [roBERT_ro_model, roBERT_ro_tokenizer],
                  'wordlist':wordlist_roBERT, 'text':en2ro1984_roBERT},
    }

    # # warnings.filterwarnings("ignore")
    # # print('Generate dictionary training data')
    # singleword_pairs = pd.read_csv('word_lists/en_ro_singleword_500.csv', sep='\t', encoding='utf-8')
    # df_dict = singleword_pairs.to_dict(orient='series')
    # en_words, ro_words = [df_dict[lang].to_list() for lang in ('en', 'ro')]
    # X_wordlist = get_vectors_from_wordlist(en_words, word2vec.get_vector, [word2vec_en])
    # Y_wordlist = get_vectors_from_wordlist(ro_words, word2vec.get_vector, [word2vec_ro])
    # X_wordlist = get_vectors_from_wordlist(en_words, bert_get_word_vector,
    #                                        [base_bert_en_model, base_bert_en_tokenizer])
    # Y_wordlist = get_vectors_from_wordlist(ro_words, bert_get_word_vector,
    #                                        [bert1_ro_model, bert1_ro_tokenizer])
    # X_wordlist, Y_wordlist = training_set_exclude_nulls(X_wordlist, Y_wordlist)

    result_dict = {}

    for combo_name in ('RoBERT',):
        combo_data = combo_dict[combo_name]
        print(f'*********** {combo_name} ****************')
        wordlist_data = combo_data['wordlist']
        X_wordlist = wordlist_data['en']
        Y_wordlist = wordlist_data['ro']

        text_pair_data = combo_data['text']
        metricFn = None

        results_wordlist = evaluate_adaptive_estimator(text_pair_data, LinearRegression,
                                                       X_wordlist, Y_wordlist, metricFn)
        print('With wordlist: ', sum(results_wordlist)/len(results_wordlist))
        results_blank_slate = evaluate_adaptive_estimator(text_pair_data, LinearRegression,
                                                          None, None, metricFn)
        print('With blank slate: ', sum(results_blank_slate)/len(results_blank_slate))
        result_dict[combo_name] = {'wordlist':results_wordlist, 'blank_slate': results_blank_slate}

