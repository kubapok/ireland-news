import datetime
from config import LABELS_DICT

with open('../test-A/in.tsv','r') as f_in, open(f'../test-A/huggingface_format_year_as_text.csv', 'w') as f_hf:
    f_hf.write('text\tyear_cont\tdate\tday_of_year\tday_of_month\tmonth\tyear\tweekday\tlabel\n')
    for line_in in f_in:
        year_cont, date, text = line_in.rstrip('\n').split('\t')
        d = datetime.datetime.strptime(date,"%Y%m%d")
        day_of_year = str(d.timetuple().tm_yday)
        day_of_month = str(d.day)
        month = str(d.month)
        year = str(d.year)
        weekday = str(d.weekday())
        day_of_year = str(d.timetuple().tm_yday)
        text = 'year: ' + year + ' month: ' + month + ' day: ' + day_of_month + ' weekday: ' + weekday + ' ' + text
        f_hf.write(text +'\t' +year_cont +'\t'+ date + '\t' +  day_of_year + '\t' + day_of_month + '\t' + month + '\t' + year + '\t' + weekday + '\t' +  str('0') + '\n')


for dataset in 'train', 'dev-0':
    with open(f'../{dataset}/in.tsv') as f_in, open(f'../{dataset}/expected.tsv') as f_exp, open(f'../{dataset}/huggingface_format_year_as_text.csv','w') as f_hf:
        f_hf.write('text\tyear_cont\tdate\tday_of_year\tday_of_month\tmonth\tyear\tweekday\tlabel\n')
        for line_in, line_exp in zip(f_in, f_exp):
            label = str(LABELS_DICT[line_exp.rstrip('\n')])
            year_cont,date,text = line_in.rstrip('\n').split('\t')
            d = datetime.datetime.strptime(date,"%Y%m%d")
            day_of_year = str(d.timetuple().tm_yday)
            day_of_month = str(d.day)
            month = str(d.month)
            year = str(d.year)
            weekday = str(d.weekday())
            day_of_year = str(d.timetuple().tm_yday)
            text = 'year: ' + year + ' month: ' + month + ' day: ' + day_of_month + ' weekday: ' + weekday + ' ' + text
            f_hf.write(text +'\t' +year_cont +'\t'+ date + '\t'+ day_of_year + '\t' + day_of_month + '\t' + month + '\t' + year + '\t' + weekday + '\t'  +  label + '\n')

