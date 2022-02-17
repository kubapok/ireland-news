from config import LABELS_DICT

with open('../test-A/in.tsv','r') as f_in, open(f'../test-A/huggingface_format.tsv', 'w') as f_hf:
    f_hf.write('text\n')
    for line_in in f_in:
        _,_, text = line_in.split('\t')
        f_hf.write(text)


for dataset in 'train_100k', 'dev-0':
    with open(f'../{dataset}/in.tsv') as f_in, open(f'../{dataset}/expected.tsv') as f_exp, open(f'../{dataset}/huggingface_format.csv','w') as f_hf:
        f_hf.write('text\tlabel\n')
        for line_in, line_exp in zip(f_in, f_exp):
            label = LABELS_DICT[line_exp.rstrip('\n')]
            _,_,text = line_in.rstrip('\n').split('\t')
            f_hf.write(text +'\t'+ str(label) + '\n')

