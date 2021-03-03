from collections import defaultdict

import nltk
import numpy as np
import random
import string
import torch
from fitbert import FitBert
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
from tqdm import tqdm
from utils import color_print_top_words

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Initialize BERT vocabulary...')
bert_tokenizer = BertTokenizer(vocab_file='data/BERT_model_reddit/vocab.txt')
print('Initialize BERT model...')
bert_model = BertForMaskedLM.from_pretrained('data/BERT_model_reddit').to(device)
bert_model.eval()


def MLM(sgs, drug_formal, thres=1, skip_flag=1):
    def to_bert_input(tokens, bert_tokenizer):
        token_idx = torch.tensor(bert_tokenizer.convert_tokens_to_ids(tokens))
        sep_idx = tokens.index('[SEP]')
        segment_idx = token_idx * 0
        segment_idx[(sep_idx + 1):] = 1
        mask = (token_idx != 0)
        return token_idx.unsqueeze(0).to(device), segment_idx.unsqueeze(0).to(device), mask.unsqueeze(0).to(device)

    def single_MLM(message):
        MLM_k = 50  # drug is 50,
        tokens = bert_tokenizer.tokenize(message)
        if len(tokens) == 0:
            return []
        if tokens[0] != CLS:
            tokens = [CLS] + tokens
        if tokens[-1] != SEP:
            tokens.append(SEP)
        token_idx, segment_idx, mask = to_bert_input(tokens, bert_tokenizer)
        with torch.no_grad():
            logits = bert_model(token_idx, segment_idx, mask, masked_lm_labels=None)
        logits = logits.squeeze(0)
        probs = torch.softmax(logits, dim=-1)

        for idx, token in enumerate(tokens):
            if token == MASK:
                topk_prob, topk_indices = torch.topk(probs[idx, :], MLM_k)
                topk_tokens = bert_tokenizer.convert_ids_to_tokens(topk_indices.cpu().numpy())

        out = [[topk_tokens[i], float(topk_prob[i])] for i in range(MLM_k)]
        return out

    PAD, MASK, CLS, SEP = '[PAD]', '[MASK]', '[CLS]', '[SEP]'
    MLM_score = defaultdict(float)
    temp = sgs if len(sgs) < 10 else tqdm(sgs)
    skip_total_num = 0
    good_sgs = []
    for sgs_i in temp:
        top_words = single_MLM(sgs_i)
        seen_target = 0
        for target in drug_formal:
            if target in [x[0] for x in top_words[:thres]]:
                seen_target += 1
        if skip_flag == 1 and seen_target < 2:
            skip_total_num += 1
            continue
        good_sgs.append(sgs_i)
        for j in top_words:
            if j[0] in string.punctuation:
                continue
            if j[0] in stopwords.words('english'):
                continue
            if j[0] in drug_formal:
                continue
            if j[0] in ['drug', 'drugs']:
                continue
            if j[0][:2] == '##':  # the '##' by BERT indicates that is not a word.
                continue
            MLM_score[j[0]] += j[1]
        print(sgs_i)
        print([x[0] for x in top_words[:20]])
    out = sorted(MLM_score, key=lambda x: MLM_score[x], reverse=True)
    out_tuple = [[x, MLM_score[x]] for x in out]
    if len(sgs) >= 10:
        print('skip_total_num is {:d}/{:d} = {:.2f}%'.format(skip_total_num, len(sgs), float(skip_total_num)/len(sgs)*100))
    return out, out_tuple, good_sgs


def multi_MLM(sgs, drug_formal, top_words, dataset, rank_model):
    def read_phrase_candidates(fpath, thres):
        phrase_cand = []
        with open(fpath, 'r') as fin:
            for line in fin:
                temp = line.split('\t')
                if float(temp[0]) < thres:
                    break
                temp_2 = temp[1].split()
                if len(temp_2) > 2:
                    continue
                phrase_cand.append(temp[1].strip().lower())
        return phrase_cand

    def filter_phrase(phrase_cand, top_words):
        out = []
        top_words = set(top_words)
        block_words = set([y.lower() for y in drug_formal])
        # block_words = set([y.lower() for y in drug_formal + ['prescription', 'vendor', 'pain', 'medical', 'synthetic', 'quality']])
        for phrase_i in phrase_cand:
            temp = [x.lower() for x in phrase_i.split()]
            if not any(x in top_words for x in temp):  # Euphemisms must contain top 1-gram.
                continue
            if any(x in block_words for x in temp):  # Euphemisms should not contain drug formal names and other block names.
                continue
            out.append(phrase_i.lower())
        return out

    def rank_by_spanbert(phrase_cand, sgs, drug_formal):
        from transformers import BertForMaskedLM, BertTokenizer
        bert_tokenizer = BertTokenizer.from_pretrained('data/BERT_model_reddit/vocab.txt')
        bert_model = BertForMaskedLM.from_pretrained('data/BERT_model_reddit').to(device)
        fb = FitBert(model=bert_model, tokenizer=bert_tokenizer, mask_token='[MASK]')
        MLM_score = defaultdict(float)
        temp = sgs if len(sgs) < 10 else tqdm(sgs)
        for sgs_i in temp:
            if not any(x in sgs_i for x in drug_formal + ['drug']):
                continue
            temp = fb.rank_multi(sgs_i, phrase_cand + ['cbd oil', 'hash oil', 'charlie horse', 'lunch money'])
            scores = [x / max(temp[1]) for x in temp[1]]
            scores = fb.softmax(torch.tensor(scores).unsqueeze(0)).tolist()[0]
            top_words = [[temp[0][i], scores[i]] for i in range(min(len(temp[0]), 50))]
            for j in top_words:
                if j[0] in string.punctuation:
                    continue
                if j[0] in stopwords.words('english'):
                    continue
                if j[0] in drug_formal:
                    continue
                if j[0] in ['drug', 'drugs']:
                    continue
                if j[0][:2] == '##':  # the '##' by BERT indicates that is not a word.
                    continue
                MLM_score[j[0]] += j[1]
            print(sgs_i)
            print([x[0] for x in top_words[:20]])
        out = sorted(MLM_score, key=lambda x: MLM_score[x], reverse=True)
        out_tuple = [[x, MLM_score[x]] for x in out]
        return out, out_tuple

    def rank_by_word2vec(phrase_cand):
        import time
        from gensim.models.word2vec import LineSentence
        from gensim.models import KeyedVectors, Word2Vec
        def train_word2vec_embed(sentences, new_text_file, embed_fn, ft=10, vec_dim=50, window=8):
            with open(new_text_file, 'w') as fout:
                for i in sentences:
                    fout.write(i + '\n')
            sentences = LineSentence(new_text_file)
            sent_cnt = 0
            for sentence in sentences:
                sent_cnt += 1
            print("# of sents: {}".format(sent_cnt))
            start = time.time()
            model = Word2Vec(sentences, min_count=ft, size=vec_dim, window=window, iter=10, workers=30)
            print("embed train time: {}s".format(time.time() - start))
            model.wv.save_word2vec_format(embed_fn, binary=False)
            return model

        new_text = []
        fname = './data/' + dataset + '.txt'
        num_lines = sum(1 for line in open(fname, 'r'))
        with open(fname, 'r') as fin:
            for line in tqdm(fin, total=num_lines):
                for j in phrase_cand:
                    line = line.replace(j, '_'.join(j.split()))
                new_text.append(line.strip())

        embed_file = './data/embeddings/embeddings_' + dataset + '.txt'
        new_text_file = './data/embeddings/new_' + dataset + '.txt'
        word2vec_model = train_word2vec_embed(new_text, new_text_file, embed_file)
        emb_dict = KeyedVectors.load_word2vec_format(embed_file, binary=False, limit=20000)
        target_vector = []
        seq = []
        for i, seed in enumerate(drug_formal):
            if seed in emb_dict:
                target_vector.append(emb_dict[seed])
                seq.append(i)
        target_vector = np.array(target_vector)
        target_vector_ave = np.sum(target_vector, 0) / len(target_vector)
        out = [' '.join(x[0].split('_')) for x in word2vec_model.wv.similar_by_vector(target_vector_ave, topn=len(emb_dict.vocab)) if '_' in x[0] and not any(y in drug_formal for y in x[0].split('_'))]
        return out, []

    phrase_cand = read_phrase_candidates('./data/phrase/AutoPhrase_' + dataset + '.txt', 0.7)
    phrase_cand = filter_phrase(phrase_cand, top_words)
    if rank_model in ['epd', 'word2vec']:
        phrase_cand, _ = rank_by_word2vec(phrase_cand)
    if rank_model in ['epd', 'epd-rank-all']:
        phrase_cand, _ = rank_by_spanbert(phrase_cand, sgs, drug_formal)
    return phrase_cand, []


def evaluate_detection(top_words, drug_euphemism):
    correct_list = []
    correct_list_upper = []
    drug_euphemism_upper = set([y for x in drug_euphemism for y in x.split()])
    for i, x in enumerate(top_words):
        correct_list.append(1 if x in drug_euphemism else 0)
        correct_list_upper.append(1 if any(y in drug_euphemism_upper for y in x.split()) else 0)
    topk_precision_list = []
    cummulative_sum = 0
    topk_precision_list_upper = []
    cummulative_sum_upper = 0
    for i in range(0, len(correct_list)):
        cummulative_sum += correct_list[i]
        topk_precision_list.append(cummulative_sum/(i+1))
        cummulative_sum_upper += correct_list_upper[i]
        topk_precision_list_upper.append(cummulative_sum_upper/(i+1))

    axes = plt.gca()
    axes.set_ylim([0, 0.7])
    plot_length = 1000
    plt.plot(range(min(plot_length, len(topk_precision_list))), topk_precision_list[:plot_length], label="Lower")
    # plt.plot(range(min(plot_length, len(topk_precision_list_upper))), topk_precision_list_upper[:plot_length], label="Upper")
    plt.xlabel('Top-k')
    plt.ylabel('Precision')
    plt.title('Euphemism Detection')
    # plt.legend()
    # plt.savefig('./results/Precision_' + str(datetime.datetime.now()) + '.png', bbox_inches='tight')
    # plt.show()
    plt.close()

    axes = plt.gca()
    axes.set_ylim([0, 300])
    plt.plot(range(min(plot_length, len(topk_precision_list))), [x*(i+1) for i, x in enumerate(topk_precision_list[:plot_length])], label="Lower")
    plt.plot(range(min(plot_length, len(topk_precision_list_upper))), [x*(i+1) for i, x in enumerate(topk_precision_list_upper[:plot_length])], label="Upper")
    plt.plot(range(min(plot_length, len(topk_precision_list_upper))), range(min(plot_length, len(topk_precision_list_upper))), 'r--', label="Perfect")
    plt.xlabel('Top-k')
    plt.ylabel('Cummulative Sum')
    plt.title('Euphemism Detection')
    plt.legend()
    # plt.savefig('./results/CSum_' + dataset + str(datetime.datetime.now()) + '.png', bbox_inches='tight')
    # plt.show()
    plt.close()
    for topk in [10, 20, 30, 40, 50, 60, 80, 100, 200, 500, 1000, 2000, 3000, 5000, 10000, 15000, 20000, 25000, 30000]:
        if topk < len(topk_precision_list):
            print('Top-{:d} precision is ({:.2f}, {:.2f})'.format(topk, topk_precision_list[topk-1], topk_precision_list_upper[topk-1]))
    return topk_precision_list


def euphemism_detection(drug_formal, dataset, all_text, drug_euphemism, rank_model, skip, multi):
    print('*' * 80)
    print('[util.py] ' + dataset + '_drug_formal: ', end='')
    print(drug_formal)
    MASK = ' [MASK] '
    print('[util.py] Extracting skip-grams for drug_formal...')
    masked_sentence = []
    for target in drug_formal:
        for i in tqdm(all_text):
            temp = nltk.word_tokenize(i)
            if target not in temp:
                continue
            temp_index = temp.index(target)
            masked_sentence += [' '.join(temp[: temp_index]) + MASK + ' '.join(temp[temp_index + 1:])]
    random.shuffle(masked_sentence)
    masked_sentence = masked_sentence[:2000]
    print('[util.py] Generating top candidates...')
    if multi == 0:
        top_words, top_words_tuple, _ = MLM(masked_sentence, drug_formal, thres=5, skip_flag=skip)
    else:
        ini_top_words, _, good_masked_sentence = MLM(masked_sentence, drug_formal, thres=5, skip_flag=skip)
        top_words, top_words_tuple = multi_MLM(good_masked_sentence, drug_formal, ini_top_words[:100], dataset, rank_model)

    print('Candidates: ')
    color_print_top_words(top_words, drug_euphemism)
    evaluate_detection(top_words, drug_euphemism)
    return top_words


