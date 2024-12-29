
import transformers
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer
from transformers import TrainingArguments

from datasets import load_dataset
from datasets import Dataset

import utils
from eval_adnotation import ParallelTexts

model_source = "google-bert/bert-base-multilingual-cased"

# tokenizer = AutoTokenizer.from_pretrained(model_source)

en2ro_source = "./1984_pickled/en2ro-1984-word2vec-latest.p"

lang_names = {'ro': 'Romanian', 'en' : 'English'}

def text_pairs_to_qa_dict(**kwargs): # yields dict
    text_pairs = kwargs.get('text_pairs')
    if not text_pairs:
        raise Exception('No textpairs provided!')
    for tp in text_pairs:
        if not tp.linked_tok_ids:
            continue
        for source_index, target_index in tp.linked_tok_ids.items():
            if not isinstance(target_index, int) or target_index < 0:
                continue
            qa_id = '/'.join([str(tp.meta.get(k)) for k in ('title', 'source', 'target', 'index')]
                              + [str(source_index)])
            qa_title = str(tp.meta.get('title'))
            src_lang, target_lang = lang_names[tp.meta['source']], lang_names[tp.meta.get['target']]
            answer_start = 0
            # build context
            context = f"In {src_lang}> :" + tp.source.get_word_list()) + "\n"
            context += f"<{target_lang}>"
            for i, word in enumerate(tp.target.get_word_list()):
                if i == target_index:
                    answer_start = len(context)+1
                context += (" " + word)
            context += "\n"
            question = f'What is the Romanian equivalent of word number {source_index+1}, "{tp.source.get_word(source_index)}"?'
            answer_text = tp.target.get_word(target_index)
            data = {'id':qa_id, 'title':qa_title, 'context':context, 'question':question,
                    'answers':{'text':[answer_text], 'answer_start':[answer_start]}}
            yield data

if __name__ == "__main__":
    print(f'Loading model {model_source}')
    # model = AutoModelForQuestionAnswering.from_pretrained(model_source)
    # tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = utils.p_load('./bert_lang_model/bert_QA_model.p')
    tokenizer = utils.p_load('./bert_lang_model/bert_QA_tokenizer.p')

    # text_pairs = ParallelTexts.from_pickle(en2ro_source)
    print('Loading dataset')
    dataset : Dataset = utils.p_load('./1984_pickled/qa_dataset.p')
    ds_dict = dataset.train_test_split(0.2)
    args = TrainingArguments(
        "1984-QA-bert_multilanguage",
        evaluation_strategy="no",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=True,
        push_to_hub=False, #!
    )
    print('Training...')
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_dict['train'],
        eval_dataset=ds_dict['test'],
        tokenizer=tokenizer,
    )
    trainer.train()
