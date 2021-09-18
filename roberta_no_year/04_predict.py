import pickle
from config import LABELS_LIST, MODEL
from transformers import AutoTokenizer
from tqdm import tqdm

device = 'cpu'


from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained('test_trainer/checkpoint-80/')
tokenizer = AutoTokenizer.from_pretrained(MODEL)

for dataset in ('dev-0', 'test-A'):
    with open(f'../{dataset}/in.tsv') as f_in, open(f'../{dataset}/out.tsv','w') as f_out:
        for line_in in tqdm(f_in, total=150_000):
            _,_, text = line_in.split('\t')
            text = text.rstrip('\n')
            inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = model(**inputs)
            probs = outputs[0].softmax(1)
            prediction = LABELS_LIST[probs.argmax(1)]
            f_out.write(prediction + '\n')

