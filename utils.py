import time
from collections import defaultdict
from tqdm import tqdm


''' read files '''


def read_all_text(fname, drug_formal):
    print('[read_data.py] Reading data...')
    start = time.time()
    all_text = []
    num_lines = sum(1 for line in open(fname, 'r'))
    with open(fname, 'r') as fin:
        for line in tqdm(fin, total=num_lines):
            temp = line.split()
            if any(ele in temp for ele in drug_formal) and len(line) <= 150:
                all_text.append(line.strip())
    print('[read_data.py] Finish reading data using %.2fs' % (time.time() - start))
    return all_text


def read_drugs(fname_euphemism_answer):
    euphemism_answer = defaultdict(list)
    with open(fname_euphemism_answer, 'r') as fin:
        for line in fin:
            ans = line.split(':')[0].strip().lower()
            for i in line.split(':')[1].split(';'):
                euphemism_answer[i.strip().lower()].append(ans)
    drug_euphemism = sorted(list(set([x[0] for x in euphemism_answer.items()])))
    drug_formal = sorted(list(set([y for x in euphemism_answer.items() for y in x[1]])))
    return euphemism_answer, drug_euphemism, drug_formal


''' print functions '''


class print_color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


def color_print_top_words(top_words, drug_euphemism):
    drug_euphemism_upper = set([y for x in drug_euphemism for y in x.split()])
    for i in top_words[:100]:
        if i in drug_euphemism:
            print(print_color.BOLD + print_color.PURPLE + i + print_color.END, end=', ')
        elif i in drug_euphemism_upper:
            print(print_color.UNDERLINE + print_color.PURPLE + i + print_color.END, end=', ')
        elif any(x in drug_euphemism_upper for x in i.split()):
            for x in i.split()[:-1]:
                if x in drug_euphemism_upper:
                    print(print_color.UNDERLINE + print_color.PURPLE + x + print_color.END, end=' ')
                else:
                    print(x, end=' ')
            x = i.split()[-1]
            if x in drug_euphemism_upper:
                print(print_color.UNDERLINE + print_color.PURPLE + x + print_color.END, end=', ')
            else:
                print(x, end=', ')
        else:
            print(i, end=', ')
    print()
