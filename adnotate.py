import numpy as np
from sklearn.linear_model import LinearRegression

import utils
import word2vec


prelim_training_data : dict[str, list[np.ndarray]]= None
chunk_tokens_annotation : list[dict] = None

def load_work(suffix = "-latest"):
    global prelim_training_data
    global chunk_tokens_annotation
    prelim_training_data = utils.p_load(f'./en_ro450_singlword_vec2word{suffix}.p')
    chunk_tokens_annotation = utils.p_load(f'./en_ro_1984_noun_options-annot{suffix}.p')


print('Loading data')

load_work()

X : list[np.ndarray] = prelim_training_data['en']
Y : list[np.ndarray] = prelim_training_data['ro']

print(len(X), len(Y))

# 1984 corpus data
token_data : list[dict[str, list[dict]]] = utils.p_load('trash/en_ro_1984_paired_tokens_noun_vecs.p')
# print('Adding existing annotation')
# if len(token_data) != len(chunk_tokens_annotation):
#     raise Exception('Unequal len for raw and annotation data!')
# for i in range(len(token_data)):
#     chunk_raw = token_data[i]
#     annotated_chunk = chunk_tokens_annotation[i]
#     if 'annotations' in annotated_chunk:
#         for en_id, ro_id in annotated_chunk['annotations'].items():
#             if not isinstance(ro_id, int):
#                 continue # has no match
#             en_token_vector = chunk_raw['en'][en_id].get('vector')
#             if en_token_vector is None:
#                 print(f'Error! Chunk {i}, english, token {en_id} has no vector!')
#                 continue
#             if ro_id < 0: # no choice
#                 continue
#             ro_token_vector = chunk_raw['ro'][ro_id].get('vector')
#             if ro_token_vector is None:
#                 print(f'Error! Chunk {i}, romanian, token {ro_id} has no vector!')
#                 continue
#             X.append(en_token_vector)
#             Y.append(ro_token_vector)
print('Generating model')
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

def save_work(suffix = "-latest"):
    utils.p_save(prelim_training_data, f'./en_ro450_singlword_vec2word{suffix}.p')
    utils.p_save(chunk_tokens_annotation, f'./en_ro_1984_noun_options-annot{suffix}.p')

if __name__ == "__main__":

    for i, annot_chunk in enumerate(chunk_tokens_annotation):
        chunk_with_data = token_data[i]
        en_text = annot_chunk['en']
        ro_text = annot_chunk['ro']
        en_words, ro_words = [text.split(' ') for text in (en_text, ro_text)]
        options = annot_chunk['options']
        option_list = list(options.items())
        if 'annotations' not in annot_chunk:
            annot_chunk['annotations'] = {}
        en_noun_count = 0
        break_flag = False
        back_up_flag = False
        while en_noun_count < len(option_list):
            print(f'Chunk {i+1} of {len(chunk_tokens_annotation)}, noun {en_noun_count+1} of {len(option_list)}:')
            en_id, ro_options = option_list[en_noun_count]
            if not ro_options:
                en_noun_count += 1
                continue
            if en_id in annot_chunk['annotations'] and not back_up_flag:
                en_noun_count += 1
                continue
            back_up_flag = False
            print('\t'+en_text)
            print('\t'+ro_text)
            en_word = en_words[en_id]
            default_id = ro_options[0] if ro_options else 'NONE'
            # I'm going to redo the ro_options with the latest regressor

            # ro_words = [ro_tokens[ro_id] for ro_id in ro_options]
            ro_ids_vecs = [{'id':ro_id,
                            'vector':chunk_with_data['ro'][ro_id]['vector']
                           } for ro_id in ro_options
                if 'vector' in chunk_with_data['ro'][ro_id]]

            en_token_vector = chunk_with_data['en'][en_id].get('vector')
            if en_token_vector is None:
                print(f'Error! Chunk {i}, english, token {en_id} has no vector!')
            elif ro_ids_vecs:
                predicted_ro_vector = lin_reg.predict([en_token_vector])[0]
                ro_ids_vecs = word2vec.find_best_match(predicted_ro_vector, ro_ids_vecs)
                default_id = ro_ids_vecs[0]['id']

            print(f'{en_word} = {ro_words[default_id]}')
            option_dict = {'' : default_id}
            ro_options.sort()
            if len(ro_options) > 1:
                option_dict.update({str(i):ro_id for i, ro_id in enumerate(ro_options)})
            option_dict.update({'-1':'back 1', '-2':'exit'})
            for i, (key, ro_id) in enumerate(option_dict.items()):
                print(f'\t{repr(key)}={ro_words[ro_id] if isinstance(ro_id, int) and ro_id >= 0  else ro_id}', end='')
                if i%5 == 0:
                    print()
            choice = input()
            if choice == '-1':
                if en_noun_count > 0:
                    en_noun_count -= 1
                back_up_flag = True
                continue
            if choice == '-2':
                break_flag = True
                break
            if choice == '':
                choice = default_id if len(ro_words)>0 else -1
            elif choice in option_dict:
                choice = option_dict[choice]
            annot_chunk['annotations'][en_id] = choice
            # now redo the predictor with the new choice
            if isinstance(choice, int):
                ro_token_vector = chunk_with_data['ro'][choice].get('vector')
                if ro_token_vector is None:
                    print(f'No vector for ro, chunk {i}, token {choice}!')
                else:
                    X.append(en_token_vector)
                    Y.append(ro_token_vector)
                    lin_reg = LinearRegression()
                    lin_reg.fit(X, Y)
            en_noun_count += 1
            # end while loop
        if break_flag:
            break



