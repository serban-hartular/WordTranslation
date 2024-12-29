import numpy as np
from sklearn.linear_model import LinearRegression

import eval_adnotation
import utils
import word2vec

from eval_adnotation import ParallelTexts, TextPair, TextChunk, get_training_set_from_parallel_texts

print('Loading data...')
en2ro_manual = ParallelTexts.from_pickle('./1984_pickled/en2ro-1984-word2vec-justmanual.p')

print('Getting and training model from existing...')
X, Y = get_training_set_from_parallel_texts(en2ro_manual)

translator = LinearRegression()
translator.fit(X, Y)

def save_work(suffix : str = 'latest'):
    en2ro_manual.pickle(f'./1984_pickled/en2ro-1984-word2vec-{suffix}.p')

selected_option = None

for tp_index, tp in enumerate(en2ro_manual):
    if tp.linked_tok_ids: # already annotated
        continue
    source = tp.source
    target = tp.target
    if not source.candidates or not target.candidates:
        continue
    target_vectors = list(zip(target.candidates, target.vectors))
    for src_candidate_index, (src_candidate, src_vector) in enumerate(zip(source.candidates, source.vectors)):
        print(f'Chunk {tp_index}, candidate word {src_candidate_index+1}')
        # print texts
        source_words = ["*"+word+"*" if i == src_candidate else word for i, word in enumerate(source.get_word_list())]
        print('SOURCE:\t' + ' '.join(source_words))
        print('TARGET:\t' + ' '.join(target.get_word_list()))
        # calculate default
        translated_vector = translator.predict([src_vector])[0]
        index_scores = eval_adnotation.guess_closest_match(translated_vector, target_vectors)
        default_index = index_scores[0][0]
        print(f'PICK {source.get_word(src_candidate)} = ...{target.get_word(default_index)}')
        # display options
        options = ({'default':default_index} |
                   {str(i+1):c_id for i, c_id in enumerate(target.candidates)})
        for opt_index, (in_nr, option) in enumerate(options.items()):
            print(f'{in_nr} = {target.get_word(option)}', end='\t')
            if opt_index % 5 == 4:
                print()
        print('-1 : exit')
        selected_option = input()
        if selected_option == "-1":
            break
        if selected_option == "": selected_option = "default"
        selected_target_vector = None
        if selected_option in options:
            selected_option = options[selected_option]
            selected_target_vector = target.get_vector(selected_option)
            X.append(src_vector)
            Y.append(selected_target_vector)
            translator = LinearRegression()
            translator.fit(X, Y)
        tp.linked_tok_ids[src_candidate] = selected_option
    if selected_option == "-1":
        break
