from __future__ import annotations

import random

import utils
from transformers import BertTokenizer, BertModel
import numpy as np


def get_sentence_embeddings(sentence_list : list[str], model : BertModel, tokenizer : BertTokenizer) -> list[np.ndarray]:
    encoding = tokenizer(sentence_list, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**encoding)
    sentence_embeddings = [pt_vectors[0].detach().numpy() for pt_vectors in outputs.last_hidden_state]
    return sentence_embeddings

def sentence_embbeding_dict(sentence_dict : dict[str|int, str],
                            model : BertModel, tokenizer : BertTokenizer,
                            window_size : int = 1) -> dict[str|int, np.ndarray]:
    sentence_dict = list(sentence_dict.items())
    vector_dict = {}
    for i, (sent_id, sent_text) in enumerate(sentence_dict):
        print(f'Doing {i+1} of {len(sentence_dict)}')
        start_index = 0 if i-window_size < 0 else i-window_size
        end_index = i+window_size+1
        context = [t[1] for t in sentence_dict[start_index:end_index]] # get texts
        sentence_vectors = get_sentence_embeddings(context, model, tokenizer)
        current_vector = sentence_vectors[i-start_index]
        vector_dict[sent_id] = current_vector
    return vector_dict

if __name__ == "__main__":
    print('Loading data...')
    en_sentences: dict[str, str] = utils.p_load('./1984_texts/xml_en_sents.p')
    ro_sentences: dict[str, str] = utils.p_load('./1984_texts/xml_ro_sents.p')
    chunk_pairs: list[tuple[list, list]] = utils.p_load('./1984_texts/en_ro_chunk_pairs.p')

    model_source = "google-bert/bert-base-multilingual-cased"

    multilang_tokenizer = BertTokenizer.from_pretrained(model_source)
    multilang_model = BertModel.from_pretrained(model_source)

    # en_sent_vecs_w1 = sentence_embbeding_dict(en_sentences, multilang_model, multilang_tokenizer)
    en_sent_vecs_w1 : dict[str, np.ndarray] = utils.p_load('./1984_texts/en_vecs_multilang_w1.p')
    ro_sent_vecs_w1 : dict[str, np.ndarray] = utils.p_load('./1984_texts/ro_vecs_multilang_w1.p')

    print('Generating training vectors')
    ID_data = []
    # generate matching and non-matching ids
    for i, chunk in enumerate(chunk_pairs):
        before = chunk_pairs[i-1] if i > 0 else ([], [])
        after = chunk_pairs[i+1] if i+1 < len(chunk_pairs) else ([], [])
        no_match = (before[0]+after[0], before[1]+after[1])
        # first do matching data
        for en_id in chunk[0]:
            for ro_id in chunk[1]:
                ID_data.append((en_id, ro_id, 1))
        # now do non-matching data
        for en_id in chunk[0]:
            for ro_id in no_match[1]:
                ID_data.append((en_id, ro_id, 0))
        for ro_id in chunk[1]:
            for en_id in no_match[0]:
                ID_data.append((en_id, ro_id, 0))
    # now do vector data
    X = []
    y = []
    for en_id, ro_id, result in ID_data:
        X.append(np.concatenate([en_sent_vecs_w1[en_id], ro_sent_vecs_w1[ro_id]]))
        y.append(result)
    # split into train and test
    test_fraction = 0.25
    test_indices = random.sample(range(len(y)), int(len(y)*test_fraction))
    split_data = {'train':{'X':[], 'y':[]}, 'test':{'X':[], 'y':[]}}
    for i, (vec, result) in enumerate(zip(X, y)):
        label = 'test' if i in test_indices else 'train'
        split_data[label]['X'].append(vec)
        split_data[label]['y'].append(result)


