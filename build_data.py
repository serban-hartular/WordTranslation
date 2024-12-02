
import eval_adnotation
import utils
import word2vec
import bert_embeddings

print('Hello!')

raw_tps = utils.p_load('./trash/en_ro_1984_paired_tokens_noun_vecs.p')
raw_annot = utils.p_load('./trash/en_ro_1984_noun_options-annot-latest.p')

en_ro_1984_parallel = eval_adnotation.ParallelTexts()
for i, raw_rec in enumerate(raw_tps):
    chunks = []
    for lang in ('en', 'ro'):
        tok_list = raw_rec[lang]
        candidates = []
        vectors = []
        for j, tok in enumerate(tok_list):
            if 'vector' not in tok:
                continue
            vector = tok.pop('vector')
            candidates.append(j)
            vectors.append(vector)
        chunks.append(eval_adnotation.TextChunk(tok_list, candidates,
                                                vectors, {'lang':lang}))
    annot = raw_annot[i]['annotations'] if 'annotations' in raw_annot[i] else {}
    tp = eval_adnotation.TextPair(chunks[0], chunks[1], annot,
                        {'title':'1984', 'source':'en', 'target':'ro', 'index':i})
    en_ro_1984_parallel.append(tp)


