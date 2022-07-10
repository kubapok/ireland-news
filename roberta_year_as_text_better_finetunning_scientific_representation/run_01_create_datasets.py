import datetime
from config import LABELS_DICT


def get_scientific_notation(i):
    s_10e0 = i % 10
    i //= 10
    s_10e1 = i  % 10 
    i //= 10
    s_10e2 = i  % 10 
    i //= 10
    s_10e3 = i  % 10 

    if s_10e3:
        return f'{s_10e3} 10e3 {s_10e2} 10e2 {s_10e1} 10e1 {s_10e0} 10e0'
    elif s_10e2:
        return f'{s_10e2} 10e2 {s_10e1} 10e1 {s_10e0} 10e0'
    elif s_10e1:
        return f'{s_10e1} 10e1 {s_10e0} 10e0'
    elif s_10e0:
        return f'{s_10e0} 10e0'
    else:
        return ''

assert get_scientific_notation(1956) == '1 10e3 9 10e2 5 10e1 6 10e0'
assert get_scientific_notation(56) == '5 10e1 6 10e0'
assert get_scientific_notation(6) == '6 10e0'




if __name__ == '__main__':
    for test_v in ('A', 'B'):
        with open(f'../test-{test_v}/in.tsv','r') as f_in, open(f'./test-{test_v}_huggingface_format_year_as_text.csv', 'w') as f_hf:
            f_hf.write('text\tyear_cont\tdate\tday_of_year\tday_of_month\tmonth\tyear\tweekday\tlabel\n')
            for line_in in f_in:
                year_cont, date, text = line_in.rstrip('\n').split('\t')
                d = datetime.datetime.strptime(date,"%Y%m%d")
                day_of_year = get_scientific_notation(d.timetuple().tm_yday)
                day_of_month = get_scientific_notation(d.day)
                month = get_scientific_notation(d.month)
                year = get_scientific_notation(d.year)
                weekday = get_scientific_notation(d.weekday())
                day_of_year = get_scientific_notation(d.timetuple().tm_yday)
                text = 'year: ' + year + ' month: ' + month + ' day: ' + day_of_month + ' weekday: ' + weekday + ' ' + text
                f_hf.write(text +'\t' +year_cont +'\t'+ date + '\t' +  day_of_year + '\t' + day_of_month + '\t' + month + '\t' + year + '\t' + weekday + '\t' +  str('0') + '\n')


    for dataset in 'train', 'dev-0':
        with open(f'../{dataset}/in.tsv') as f_in, open(f'../{dataset}/expected.tsv') as f_exp, open(f'./{dataset}_huggingface_format_year_as_text.csv','w') as f_hf:
            f_hf.write('text\tyear_cont\tdate\tday_of_year\tday_of_month\tmonth\tyear\tweekday\tlabel\n')
            for line_in, line_exp in zip(f_in, f_exp):
                label = str(LABELS_DICT[line_exp.rstrip('\n')])
                year_cont,date,text = line_in.rstrip('\n').split('\t')
                d = datetime.datetime.strptime(date,"%Y%m%d")
                day_of_year = get_scientific_notation(d.timetuple().tm_yday)
                day_of_month = get_scientific_notation(d.day)
                month = get_scientific_notation(d.month)
                year = get_scientific_notation(d.year)
                weekday = get_scientific_notation(d.weekday())
                day_of_year = get_scientific_notation(d.timetuple().tm_yday)
                text = 'year: ' + year + ' month: ' + month + ' day: ' + day_of_month + ' weekday: ' + weekday + ' ' + text
                f_hf.write(text +'\t' +year_cont +'\t'+ date + '\t'+ day_of_year + '\t' + day_of_month + '\t' + month + '\t' + year + '\t' + weekday + '\t'  +  label + '\n')

