# -*- coding: utf-8 -*-
# @Author: zsun
# @Date:   2018-10-18 15:42
from __future__ import print_function
import time
# import sys
import argparse
import random
import torch 
import torch.autograd as autograd
# import torch.nn as nn
# import torch.optim as optim
import numpy as np
from utils.metric import get_ner_fmeasure
from model.seqmodel import SeqModel
from utils.data import Data 

try:
    import cPickle as pickle
except ImportError:
    import pickle


seed_num = 42
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0 
    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1 
    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
            # return 00000
        return self.word2idx[word]  
    def __len__(self):
        return len(self.word2idx)

def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = gold_variable.size(0)
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        assert(len(pred)==len(gold))
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label


def recover_nbest_label(pred_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len, nbest): pred tag result
            mask_variable (batch_size, sent_len): mask variable
            word_recover (batch_size)
        output:
            nbest_pred_label list: [batch_size, nbest, each_seq_len]
    """
    # exit(0)
    pred_variable = pred_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = pred_variable.size(0)
    seq_len = pred_variable.size(1)
    nbest = pred_variable.size(2)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    for idx in range(batch_size):
        pred = []
        for idz in range(nbest):
            each_pred = [label_alphabet.get_instance(pred_tag[idx][idy][idz]) for idy in range(seq_len) if mask[idx][idy] != 0]
            pred.append(each_pred)
        pred_label.append(pred)
    return pred_label

@torch.no_grad()
def evaluate(data, model, name, nbest=None):
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    elif name == 'raw':
        instances = data.raw_Ids
    else:
        print("Error: wrong evaluate name,", name)
        exit(1)
    right_token = 0
    whole_token = 0
    nbest_pred_results = []
    pred_scores = []
    pred_results = []
    gold_results = []
    ## set model in eval model
    model.eval()
    batch_size = data.HP_batch_size
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num//batch_size+1
    for batch_id in range(total_batch):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size
        if end > train_num:
            end =  train_num
        instance = instances[start:end]
        if not instance:
            continue
        batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask  = batchify_with_label(instance, data.HP_gpu, True)
        # print(batch_word.size()) [7,20]
        # print(batch_features) []
        # print(len(batch_features)) 0
        # print(nbest) 1
        if nbest:
            scores, nbest_tag_seq = model.decode_nbest(batch_word,batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask, nbest)
            nbest_pred_result = recover_nbest_label(nbest_tag_seq, mask, data.label_alphabet, batch_wordrecover)
            nbest_pred_results += nbest_pred_result
            pred_scores += scores[batch_wordrecover].cpu().data.numpy().tolist()
            ## select the best sequence to evalurate
            tag_seq = nbest_tag_seq[:,:,0]
        else:
            tag_seq = model(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask)
        # print("tag:",tag_seq)
        # print("tag:",tag_seq.size()) [7,20]
        pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover)
        # print(pred_label) O S-1
        # print(gold_label)
        pred_results += pred_label
        gold_results += gold_label
    decode_time = time.time() - start_time
    speed = len(instances)/decode_time
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
    if nbest:
        return speed, acc, p, r, f, nbest_pred_results, pred_scores
    return speed, acc, p, r, f, pred_results, pred_scores

@torch.no_grad()
def batchify_with_label(input_batch_list, gpu, volatile_flag=False):
    """
        input: list of words, chars and labels, various length. [[words,chars, labels],[words,chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len)
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len)
    """
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    features = [np.asarray(sent[1]) for sent in input_batch_list]
    feature_num = len(features[0][0])
    chars = [sent[2] for sent in input_batch_list]
    labels = [sent[3] for sent in input_batch_list]
    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max()
    # word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile =  volatile_flag).long()
    word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
    # label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)),volatile =  volatile_flag).long()
    label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
    feature_seq_tensors = []
    for idx in range(feature_num):
        # feature_seq_tensors.append(autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile =  volatile_flag).long())
        feature_seq_tensors.append(autograd.Variable(torch.zeros((batch_size, max_seq_len))).long())
    # mask = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile =  volatile_flag).byte()
    mask = autograd.Variable(torch.zeros((batch_size, max_seq_len))).byte()
    for idx, (seq, label, seqlen) in enumerate(zip(words, labels, word_seq_lengths)):
        seqlen = seqlen.item()
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        # print(seqlen)
        # print(type(seqlen))
        # print([1]*seqlen)
        mask[idx, :seqlen] = torch.Tensor([1]*seqlen)
        for idy in range(feature_num):
            feature_seq_tensors[idy][idx,:seqlen] = torch.LongTensor(features[idx][:,idy])
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    for idx in range(feature_num):
        feature_seq_tensors[idx] = feature_seq_tensors[idx][word_perm_idx]

    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    ### deal with char
    # pad_chars (batch_size, max_seq_len)
    # print(max_seq_len) # tensor(20)
    # print(len(chars[idx])) # 10
    # # print((chars[idx]).type()) list
    # print((chars[idx])) # [[270], [4655, 2229], [487, 1555], [114, 17], [904, 358], [290, 163], [1093, 781], [724], [184, 128], [62]]
    # print(max_seq_len-len(chars[idx])) #tensor(10)
    max_seq_len = max_seq_len.item()
    pad_chars = [chars[idx] + [[0]] * (max_seq_len-len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    # char_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len, max_word_len)), volatile =  volatile_flag).long()
    char_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len, max_word_len))).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            # print len(word), wordlen
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size*max_seq_len,-1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size*max_seq_len,)
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        for idx in range(feature_num):
            feature_seq_tensors[idx] = feature_seq_tensors[idx].cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        mask = mask.cuda()
    return word_seq_tensor,feature_seq_tensors, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask

@torch.no_grad()
def load_model_decode(data, name):
    print("Load Model from file: ", data.model_dir)
    model = SeqModel(data)
    ## load model need consider if the model trained in GPU and load in CPU, or vice versa
    # if not gpu:
    #     model.load_state_dict(torch.load(model_dir))
    #     # model.load_state_dict(torch.load(model_dir), map_location=lambda storage, loc: storage)
    #     # model = torch.load(model_dir, map_location=lambda storage, loc: storage)
    # else:
    #     model.load_state_dict(torch.load(model_dir))
    #     # model = torch.load(model_dir)
    model.load_state_dict(torch.load(data.load_model_dir))

    print("Decode %s data, nbest: %s ..."%(name, data.nbest))
    start_time = time.time()
    speed, acc, p, r, f, pred_results, pred_scores = evaluate(data, model, name, data.nbest)
    end_time = time.time()
    time_cost = end_time - start_time
    # if data.seg:
    #     print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(name, time_cost, speed, acc, p, r, f))
    # else:
    #     print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f"%(name, time_cost, speed, acc))
    return pred_results, pred_scores

def check(file_name):
    with open('officialdoc/vocab_all5.pkl', 'rb') as f:
        vocab = pickle.load(f) # , pickle.HIGHEST_PROTOCOL
    vocab.add_word('<unk>') # 102674
    with open('officialdoc/'+file_name+'.bmes', 'r') as f:
        all_input_list = f.read().strip().replace('，\tO\n', '，\tO|').replace('。\tO\n', '。\tO|').replace('！\tO\n', '！\tO|').replace('；\tO\n', '；\tO|').split('|')#.strip() 
        print(all_input_list) # ['也\tO\n令\tO\n我\tO\n非常\tO\n激动\tO，', '\tO\n\n收获\tO\n很多\tO。']
    listt = []
    for ids, sent in enumerate(all_input_list):
        words = [tupl.split('\t')[0] for tupl in sent.split('\n')]
        # print(words) 
        for idw, word in enumerate(words):
            if vocab.__call__(word) == 102674:
                print('word')
                print(word)
                listt.append([ids, 0, idw])
    new_bmes = '\n\n' + '\n\n\n'.join(all_input_list) + '\n\n\n'
    with open('officialdoc/'+file_name+'.bmes', 'w') as f:
        f.writelines(new_bmes)
    return listt


# [[['O', 'O', 'O', 'O', 'O', 'O']], [['O', 'O', 'O']], [['O', 'O', 'O']], [['O', 'O', 'S-1', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']], [['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']], [['O', 'O', 'O', 'O']], [['O', 'O', 'O', 'O', 'O', 'O', 'O', 'S-1', 'O', 'O']]]

if __name__ == '__main__':
    # with open('officialdoc/vocab_all5.pkl', 'rb') as f:
    #     vocab = pickle.load(f) # , pickle.HIGHEST_PROTOCOL
    # vocab.add_word('<unk>')
    # print(vocab.__call__('hhh')) 102674
    # print(vocab.__call__('哈哈'))
    # print(vocab.__call__('外理'))
    # print(vocab.__call__('是'))
    config_file = 'demo.decode.config'

    data = Data()
    data.HP_gpu = torch.cuda.is_available()
    data.read_config(config_file) 
    status = data.status.lower() 
    if status == 'decode': 
        data.load(data.dset_dir)  
        data.read_config(config_file)
        result0 = check('raw')
        data.generate_instance('raw')
        decode_results, pred_scores = load_model_decode(data, 'raw')
        print('decode_results') 
        print(decode_results) 
        for unk_id in result0: 
            ids, idx, idw = unk_id 
            print(unk_id) 
            print((decode_results[ids][idx])) 
            decode_results[ids][idx][idw] = 'S-1' 
        # print(decode_results) 

        if data.nbest: 
            data.write_nbest_decoded_results(decode_results, pred_scores, 'raw') 
        else: 
            data.write_decoded_results(decode_results, 'raw') 
    else:
        print("Invalid argument! Please use valid arguments! (train/test/decode)")