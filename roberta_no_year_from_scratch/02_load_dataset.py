import pickle
from datasets import load_dataset
from transformers import AutoTokenizer, RobertaTokenizer
from config import MODEL

dataset = load_dataset('csv', sep='\t', data_files={'train': ['../train/huggingface_format.tsv'], 'test': ['../dev-0/huggingface_format.tsv']})
test_dataset = load_dataset('csv', sep='\t', data_files ='../test-A/huggingface_format.tsv')

tokenizer = RobertaTokenizer.from_pretrained(MODEL)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
test_tokenized_datasets = test_dataset.map(tokenize_function, batched=True)

train_dataset = tokenized_datasets["train"].shuffle(seed=42)
eval_dataset_full = tokenized_datasets["test"]
eval_dataset_small = tokenized_datasets["test"].select(range(2000))
test_dataset = test_tokenized_datasets["train"]

with open('train_dataset.pickle','wb') as f_p:
    pickle.dump(train_dataset, f_p)

with open('eval_dataset_small.pickle','wb') as f_p:
    pickle.dump(eval_dataset_small, f_p)

with open('eval_dataset_full.pickle','wb') as f_p:
    pickle.dump(eval_dataset_full, f_p)

with open('test_dataset.pickle','wb') as f_p:
    pickle.dump(test_dataset, f_p)
