import pandas as pd

r_out = pd.read_csv('../train/expected.tsv', names = ('class',))
most_common = r_out['class'].value_counts().idxmax()


for dataset in 'dev-0', 'test-A':
    with open(f'../{dataset}/out.tsv', 'w') as f_out, open(f'../{dataset}/in.tsv', 'r') as f_in:
        for line_in in f_in:
            f_out.write(most_common + '\n')

