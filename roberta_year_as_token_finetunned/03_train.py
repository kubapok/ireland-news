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


from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig

#model = RobertaForSequenceClassification(RobertaConfig(num_labels=7))
model = RobertaForSequenceClassification.from_pretrained('roberta-base',num_labels=7)
#model = RobertaForSequenceClassification(model.config)

from transformers import TrainingArguments


training_args = TrainingArguments("test_trainer",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        evaluation_strategy='steps',
        #eval_steps=2_000,
        #save_steps=2_000,
        eval_steps=20_000,
        save_steps=20_000,
        num_train_epochs=20,
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

#trainer.train(resume_from_checkpoint=True)
trainer.train()
trainer.save_model("./roberta-retrained")
trainer.evaluate()


eval_predictions = trainer.predict(eval_dataset_full).predictions.argmax(1)

with open('../dev-0/out.tsv', 'w') as f_out:
    for pred in eval_predictions:
        f_out.write(LABELS_LIST[pred] + '\n')

test_predictions = trainer.predict(test_dataset).predictions.argmax(1)
with open('../test-A/out.tsv', 'w') as f_out:
    for pred in test_predictions:
        f_out.write(LABELS_LIST[pred] + '\n')
