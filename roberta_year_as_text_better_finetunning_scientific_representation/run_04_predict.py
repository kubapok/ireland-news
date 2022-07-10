import pickle
from config import LABELS_LIST, MODEL

with open('train_dataset.pickle','rb') as f_p:
    train_dataset = pickle.load(f_p)

with open('eval_dataset_small.pickle','rb') as f_p:
    eval_dataset_small = pickle.load(f_p)

with open('eval_dataset_full.pickle','rb') as f_p:
    eval_dataset_full = pickle.load(f_p)

with open('test_dataset.pickle','rb') as f_p:
    test_dataset = pickle.load(f_p)


from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained('roberta-ireland').cuda()

from transformers import TrainingArguments


training_args = TrainingArguments("roberta-ireland",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        evaluation_strategy='steps',
        #eval_steps=2_000,
        #save_steps=2_000,
        eval_steps=2_000,
        save_steps=20_000,
        num_train_epochs=1,
        gradient_accumulation_steps=2,
        learning_rate = 1e-6,
        #warmup_steps=4_000,
        warmup_steps=4,
        load_best_model_at_end=True,
        )

import numpy as np
from datasets import load_metric

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


from transformers import Trainer

trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset_small,
            compute_metrics=compute_metrics,
            )


#eval_predictions = trainer.predict(eval_dataset_full).predictions.argmax(1)

#with open('../dev-0/out.tsv', 'w') as f_out:
#    for pred in eval_predictions:
#        f_out.write(LABELS_LIST[pred] + '\n')

test_predictions = trainer.predict(test_dataset).predictions.argmax(1)
with open('../test-B/out.tsv', 'w') as f_out:
    for pred in test_predictions:
        f_out.write(LABELS_LIST[pred] + '\n')

#model = AutoModelForSequenceClassification.from_pretrained('roberta-retrained/')

#for dataset in ('dev-0', 'test-A'):
#    with open(f'../{dataset}/in.tsv') as f_in, open(f'../{dataset}/out.tsv','w') as f_out:
#        for line_in in tqdm(f_in, total=150_000):
#            _,_, text = line_in.split('\t')
#            text = text.rstrip('\n')
#            inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)
#            outputs = model(**inputs)
#            probs = outputs[0].softmax(1)
#            prediction = LABELS_LIST[probs.argmax(1)]
#            f_out.write(prediction + '\n')
#
