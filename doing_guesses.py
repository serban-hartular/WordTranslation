
from eval_adnotation import *
from sklearn.linear_model import LinearRegression

p1984 = ParallelTexts.from_pickle('./1984_pickled/en2ro-1984-word2vec-justmanual.p')
X, Y = get_training_set_from_parallel_texts(p1984)
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

metric_dict = eval_parallel_text_guess_scores(p1984, lin_reg)

threshold = max(metric_dict['right'])

for tp in p1984:
    if tp.meta.get(ANNOTATIONS_KEY) == 'manual':
        continue
    guessed_annotation = text_pair_guess_annotations(tp, lin_reg, threshold)
    best_guesses = {k:v[0] for k,v in guessed_annotation.items()}
    tp.linked_tok_ids = best_guesses
    tp.meta[ANNOTATIONS_KEY] = 'guess1024'
