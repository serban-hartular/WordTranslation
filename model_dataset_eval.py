import random
import warnings

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.neural_network import MLPRegressor

import eval_adnotation
from eval_adnotation import TextPair, ParallelTexts, get_training_set_from_text_pair
from sklearn.linear_model import LinearRegression

import utils

if __name__ == "__main__":
    warnings.simplefilter("ignore")
    print('Load parallel text data')
    en2ro1984_word2vec = ParallelTexts.from_pickle('./1984_pickled/en2ro-1984-word2vec.p')
    en2ro1984_bert1 = ParallelTexts.from_pickle('./1984_pickled/en2ro-1984-bert1.p')
    en2ro1984_roBERT = ParallelTexts.from_pickle('./1984_pickled/en2ro-1984-roBERT.p')
    combo_dict = {
        'word2vec' : en2ro1984_word2vec,
        'bert1' : en2ro1984_bert1,
        'roBERT' : en2ro1984_roBERT
    }

    EstimatorClasses = [LinearRegression, MLPRegressor, KNeighborsRegressor,
                        RadiusNeighborsRegressor]

    for EstimatorClass in EstimatorClasses:
        print("********** " + EstimatorClass.__name__ + " ***********")
        for name, text_pairs in combo_dict.items():
            text_pairs = [tp for tp in text_pairs if tp.linked_tok_ids]
            tp_test = random.sample(text_pairs, int(len(text_pairs)/4))
            tp_train = [tp for tp in text_pairs if tp not in tp_test]
            X_train, Y_train = [], []
            for tp in tp_train:
                Xadd, Yadd = get_training_set_from_text_pair(tp)
                X_train.extend(Xadd)
                Y_train.extend(Yadd)
            estimator = EstimatorClass()
            estimator.fit(X_train, Y_train)
            good, total = eval_adnotation.evaluate_estimator(tp_test, estimator)
            print(name, good/total, total)
