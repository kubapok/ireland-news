import pickle
from datasets import load_dataset
from transformers import AutoTokenizer
from config import MODEL
from tqdm import tqdm

dataset = load_dataset('csv', sep='\t', data_files={'train': ['./train_huggingface_format_year_as_text.csv'], 'test': ['./dev-0_huggingface_format_year_as_text.csv']})
test_dataset = load_dataset('csv', sep='\t', data_files='./test-B_huggingface_format_year_as_text.csv')

tokenizer = AutoTokenizer.from_pretrained(MODEL)

def tokenize_function(examples):
    t = tokenizer(examples["text"], padding="max_length", truncation=True)
    return t

test_tokenized_datasets = test_dataset.map(tokenize_function, batched=True)
tokenized_datasets = dataset.map(tokenize_function, batched=True)


#for d in ('train', 'test'):
#        for i in tqdm(range(len(tokenized_datasets[d]))):
#            tokenized_datasets[d][i][column] = [tokenized_datasets[d][i][column] ] * 512 #len(tokenized_datasets[d][i]['input_ids'])
#
#d = 'train'
#for column in tqdm(('date', 'day_of_month', 'day_of_year', 'month', 'year', 'year_cont')):
#    for i in tqdm(range(len(test_tokenized_datasets[d]))):
#        test_tokenized_datasets[d][i][column] = [test_tokenized_datasets[d][i][column] ] * 512 #len(test_tokenized_datasets[d][i]['input_ids'])

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
     

