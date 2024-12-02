# import conllu_path as cp
import pickle

# def meta_contains(sentence : cp.Sentence, contains : str) -> bool:
#     if not sentence.meta:
#         return False
#     return bool([m for m in sentence.meta if contains in m])
#
# def split_by_meta(sentence_source : list[cp.Sentence], split_str : str,
#                   doc_inclusion_search = None) -> list[cp.Doc]:
#     doc_list = []
#     current_sents = []
#     for sentence in sentence_source:
#         if meta_contains(sentence, split_str):
#             doc = cp.Doc(current_sents)
#             if doc_inclusion_search is None or list(doc.search(doc_inclusion_search)):
#                 doc_list.append(cp.Doc(current_sents))
#             current_sents = []
#         current_sents.append(sentence)
#     return doc_list

def p_save(obj, filename : str):
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle)

def p_load(filename : str):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)
