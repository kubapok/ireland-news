import pickle
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration
from tqdm import tqdm
from config import LABELS_LIST

device = 'cuda'
model_path= 't5-retrained'


from transformers import AutoModelForSequenceClassification

model = T5ForConditionalGeneration.from_pretrained(model_path).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path)

for dataset in ('dev-0', 'test-A'):
    with open(f'../{dataset}/in.tsv') as f_in, open(f'../{dataset}/out.tsv','w') as f_out:
        for line_in in tqdm(f_in, total=150_000):
            _,_, text = line_in.split('\t')
            text = text.rstrip('\n')
            inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt").input_ids.to(device)
            outputs = model.generate(inputs)
            o = tokenizer.decode(outputs[0], skip_special_tokens=True)
            o = LABELS_LIST[int(o)]
            f_out.write(o + '\n')

