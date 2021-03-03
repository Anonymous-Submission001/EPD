import argparse

from detection import euphemism_detection
from utils import read_all_text, read_drugs

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="sample")
parser.add_argument('--target', type=str, default="drug")
parser.add_argument('--model', type=str, default="epd", help="choose from ['epd', 'word2vec', 'epd-rank-all']")
args = parser.parse_args()
if args.model not in ['epd', 'word2vec', 'epd-rank-all']:
    raise ValueError("model must be one of ['epd', 'word2vec', 'epd-rank-all']")

print(args)

''' read file '''
euphemism_answer, drug_euphemism, drug_formal = read_drugs('./data/answer_' + args.target + '.txt')
if args.dataset == 'sample':
    drug_formal = ['heroin', 'ecstasy', 'marijuana', 'cocaine', 'Acetaminophen and Oxycodone Combination']
all_text = read_all_text('./data/' + args.dataset + '.txt', drug_formal)


''' Euphemistic Phrase Detection'''
top_words = euphemism_detection(drug_formal, args.dataset, all_text, drug_euphemism, args.model, skip=1, multi=1)

