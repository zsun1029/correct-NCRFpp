# -*- coding: utf-8 -*-
# @Author: sz
# @Date:   2019-06-10
import re
import pickle
from random import randint 
from pypinyin import pinyin, lazy_pinyin
from Pinyin2Hanzi import DefaultDagParams
from Pinyin2Hanzi import dag
dagParams = DefaultDagParams()

# def cut_word_list(word_list):
#     len_dict = dict()
#     final_list = []
#     for i in range(len(word_list)):
#         word = word_list[i]
#         if len(word) not in len_dict.keys():
#             len_dict[len(word)] = [word] 
#         else: 
#             len_dict[len(word)].append(word) 
#     for key in len_dict.keys():
#     	if len(len_dict[key]) == 1: continue
#     	else:
#     		final_list.append(list(len_dict[key])) 
#     return final_list

# def confused_words():# 有些问题手动改下 
#     import re
#     line_list = open('G:\master\official-document\confused_words.txt', 'r', encoding='utf-8').readlines()
#     confused_words_list = []
#     confused_words_lines = []
#     for i in range(len(line_list)):
#         line = line_list[i] 
#         p = re.compile( r'“([\u4e00-\u9fa5]+)”' )
#         # word = re.search(p, line)  
#         # if word is not None:
#         # 	print(word.groups())
#         iterator = p.finditer(line)
#         word_list = []
#         for word in iterator:
#             if word.group(1) not in word_list:
#                 word_list.append(word.group(1))
#         if len(word_list) > 2: 
#             word_list = cut_word_list(word_list) 
#         elif len(word_list) == 2:  
#             word_list = [word_list]
#         else:
#             continue
#         confused_words_list.extend(word_list)
#         word_line = ''
#         for a_list in word_list:
#             for word in a_list[:-1]:
#                 word_line = word_line + word + '\t'  
#             word_line = word_line + a_list[-1] + '\n'  
#         confused_words_lines.extend(word_line) 
#     with open('G:\master\official-document\confused_words_sz.txt', 'w', encoding='utf-8') as f:
#         f.writelines(confused_words_lines)
#         return confused_words_list

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
        return self.word2idx[word]  
    def __len__(self):
        return len(self.word2idx)

def add_vocab(vocab, filename): 
    words_list = open(filename, 'r', encoding='utf-8').readlines()
    for words in words_list:
        if words == '\n': continue 
        word = words.strip().split()[0] 
        vocab.add_word(word)  
    return vocab

def change_word_a(wordid, index_list): 
    for i in range(len(index_list)):
        if wordid >= index_list[i] and wordid < index_list[i+1]:
            a = index_list[i]
            b = index_list[i+1]
            break  
    wordid = randint(a, b-1) 
    return wordid
 
def change_word_b(word, path_num=6):  
    pinyin_list = lazy_pinyin(word)
    result = dag(dagParams, pinyin_list, path_num, log=True)  # print(len(result))  10 <class 'list'>
    try:
        index = randint(0, len(result)-1) # 当result只有一位时（比如'，'），不进行更改，直接返回原word
    except:
        return word
    new_word = result[index].path 
    return new_word[0] # print(new_word)  ['疫情'] 
 
def make_new_w(new_word, thissent, num_wrongword, flag):
    new_word = new_word + '\t' + 'S-1'
    thissent = thissent + 1
    num_wrongword += 1 
    flag = 1  
    return new_word, thissent, num_wrongword, flag


def haidian_news(filename, save_name, manual_change=False, vocabid_list=None):
    index_list = [id_list[0] for id_list in vocabid_list] # print(index_list)
    new_corpus = ""
    para_list = open(filename, 'r', encoding='utf-8').readlines()
    p = re.compile( r'([\u4e00-\u9fa5]+){原文:([\u4e00-\u9fa5]+)}' )
    sent_line_list = []
    sent_wo_wrong, sent_w_wrong = [], [] 
    num_rightword, num_wrongword = 0, 0
    sent_list = []

    print('para_list_len:',len(para_list))

    for para in para_list: # len(para_list) 14306 
        para = para.strip().replace('。', '。|').replace('，', '，|')# sent_list = re.split('[。，]', para.strip()) 
        # if len(sent_list) < 2:  continue 
        sent_list.extend(para.split('|'))

    print('sent_list_len:',len(sent_list))

    with_rand_sent = []
    for sent in sent_list:  
        add = 0
        manual_change = True
        new_word_list = []
        thissent = 0
        print(sent)
        print(sent.split())
        for word in sent.strip().split():
            yuanwen = re.search(p, word)    
            if yuanwen is not None: 
                num_wrongword += 1
                word = re.sub(p, '', word) 
                new_word = word + '\t' + 'S-1' 
                thissent = thissent + 1
                num_wrongword += 1 
            else: # 该词是不带有原文错误标记的词，则随机修改某个词
                flag = 0
                if manual_change == True: 
                    print(word)
                    wordid = vocab.word2idx[word] 
                    if wordid < gate:  # 若是易混淆词，则使用易混淆词表修改
                        change_or_not = randint(0, 5)  # 0 1 2 .. 5
                        if change_or_not < 1:  
                            wordidnew = change_word_a(wordid, index_list) 
                            if wordidnew != wordid:
                                new_word, thissent, num_wrongword, flag = make_new_w(vocab.idx2word[wordidnew], thissent, num_wrongword, flag)
                    else:  # 若不是易混淆词，则
                        add = 1
                        change_or_not = randint(0, 30)  # 0 1 2 3 .. 40
                        if change_or_not < 5:   # 随便找词表里的一个词 
                            wordidnew = randint(0, vocab_num-1)  
                            new_word = vocab.idx2word[wordidnew]      
                            if new_word != word:
                                new_word, thissent, num_wrongword, flag = make_new_w(new_word, thissent, num_wrongword, flag)                           
                        elif change_or_not > 25:  # 使用汉字转拼音来修改 
                            new_word = change_word_b(word)    
                            if new_word != word:
                                new_word, thissent, num_wrongword, flag = make_new_w(new_word, thissent, num_wrongword, flag)
                if flag == 0: 
                    num_rightword += 1
                    new_word = word + '\t' + 'O'
            new_word_list.append(new_word) 
            # if thissent != 0 and thissent != 1: print(thissent, end='')
            if thissent > 4:  manual_change = False 
        new_word_list.append('') # 每一个sent创建一个空行 
        sent_line = '\n'.join(new_word_list)  
        sent_line_list.append(sent_line)  
        if thissent == 0:  
            sent_wo_wrong.append(sent_line)  
        else:
        	sent_w_wrong.append(sent_line) 
        if add == 1:
            with_rand_sent.append(sent_line)


    print('sent: right_sent:%d wrong_sent:%d'%(len(sent_wo_wrong), len(sent_w_wrong)))      
    # print('word: right_word:%d wrong_word:%d'%(num_rightword, num_wrongword))     

    for sent in with_rand_sent:   
        manual_change = True
        new_word_list = []
        thissent = 0
        for word in sent.strip().split(): 
            # 随机修改某个词
            flag = 0
            if manual_change == True: 
                wordid = vocab.word2idx[word] 
                if wordid < gate:  # 若是易混淆词，则使用易混淆词表修改
                    change_or_not = randint(0, 50)  # 0 1 2 .. 5
                    if change_or_not < 1:  
                        wordidnew = change_word_a(wordid, index_list) 
                        if wordidnew != wordid:
                            new_word, thissent, num_wrongword, flag = make_new_w(vocab.idx2word[wordidnew], thissent, num_wrongword, flag)
                else:  # 若不是易混淆词，则
                    add = 1
                    change_or_not = randint(0, 5)  # 0 1 2 3 .. 40
                    if change_or_not < 5:   # 随便找词表里的一个词 
                        wordidnew = randint(0, vocab_num-1)  
                        new_word = vocab.idx2word[wordidnew]      
                        if new_word != word:
                            new_word, thissent, num_wrongword, flag = make_new_w(new_word, thissent, num_wrongword, flag)                           
                    elif change_or_not > 25:  # 使用汉字转拼音来修改 
                        new_word = change_word_b(word)    
                        if new_word != word:
                            new_word, thissent, num_wrongword, flag = make_new_w(new_word, thissent, num_wrongword, flag)
            if flag == 0: 
                num_rightword += 1
                new_word = word + '\t' + 'O'
            new_word_list.append(new_word) 
            # if thissent != 0 and thissent != 1: print(thissent, end='')
            if thissent > 4:  manual_change = False 
        new_word_list.append('') # 每一个sent创建一个空行 
        sent_line = '\n'.join(new_word_list)  
        sent_line_list.append(sent_line)  
        sent_w_wrong.append(sent_line) 


    num_wo_wrong, num_w_wrong =  len(sent_wo_wrong), len(sent_w_wrong) 
    assert len(sent_line_list) == ( num_wo_wrong + num_w_wrong )
    # print(len(sent_line_list)) # 144701  146919    按句号&逗号划分 444368

    # sent_w_wrong.extend(more_wrong_sents(sent_w_wrong))

    print('sent: right_sent:%d wrong_sent:%d'%(num_wo_wrong, num_w_wrong))     
    # print(num_wo_wrong)          # 318493(原文)     170399(目前的文件)         234921(1/16)       
    # print(num_w_wrong)           # 125875           273969(文件4)         209447(文件sz2)
    print('word: right_word:%d wrong_word:%d'%(num_rightword, num_wrongword))     
    # print(num_rightword)         # 4779675          4432911                   4602212
    # print(num_wrongword)         # 160210           509144                   337673 
    sent_train = sent_wo_wrong[:int(num_wo_wrong*0.8)]
    sent_train.extend(sent_w_wrong[:int(num_w_wrong*0.8)]) 
    train_sent = '\n'.join(sent_train) 
    with open( save_name + '_train.txt', 'w') as f:
        f.write(train_sent)
    sent_val = sent_wo_wrong[int(num_wo_wrong*0.8):int(num_wo_wrong*0.9)]
    sent_val.extend(sent_w_wrong[int(num_w_wrong*0.8):int(num_w_wrong*0.9)]) 
    val_sent = '\n'.join(sent_val) 
    with open( save_name + '_val.txt', 'w') as f:
        f.write(val_sent)
    sent_test = sent_wo_wrong[int(num_wo_wrong*0.9):]
    sent_test.extend(sent_w_wrong[int(num_w_wrong*0.9):]) 
    test_sent = '\n'.join(sent_test) 
    with open( save_name + '_test.txt', 'w') as f:
        f.write(test_sent)

def more_wrong_sents(sent_w_wrong):
    for sent in sent_w_wrong:
        print(sent)
        print(a)




if __name__ == '__main__':

    save_name = 'officialdoc/haidian_news_sz5'
    vocabid_list_f = 'officialdoc/vocabid_list5.pkl'
    vocab_all = 'officialdoc/vocab_all5.pkl'
    # 构建公文常见混淆词表
    # confused_words_list = confused_words()  # 手动修正混淆词表后，不要再执行上一句

    # 词表加入易混淆词表的全部
    vocab = Vocabulary()  
    words_list = open('officialdoc/confused_words_sz.txt', 'r', encoding='utf-8').readlines()
    vocabid_list = []
    id_b = 0
    for words in words_list:
        id_list = []
        for word in words.strip().split('\t'):
            vocab.add_word(word)  
            id_list.append(id_b)
            id_b += 1
        vocabid_list.append(id_list) 
    with open(vocabid_list_f, 'wb') as f:
        pickle.dump(vocabid_list, f, pickle.HIGHEST_PROTOCOL)
    gate = vocab.__len__() # 832 # print(vocab.idx2word[ vocab.__len__()-1 ]) 
    # gate = 832 

    # 词表加入train\val\test的词
    vocab = add_vocab(vocab, 'officialdoc/haidian_news_sz_train.txt') 
    vocab = add_vocab(vocab, 'officialdoc/haidian_news_sz_val.txt') 
    vocab = add_vocab(vocab, 'officialdoc/haidian_news_sz_test.txt') # 92805
    vocab = add_vocab(vocab, 'wordLS.txt')  # 不知道公文里的词是否足够，所以找了一个词表自己加进去。
    with open(vocab_all, 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
 #  haidian_news()中利用拼音转化出来的word没有加到词表里

    # 读取已保存的词表
    # f = open('officialdoc/vocab_all.pkl', 'rb')
    # vocab = pickle.load(f)
    
    vocab_num = vocab.__len__()
    print( 'vocab_num:', vocab_num ) # 92805 + 25883 = 102674 

    # 带有少量修改标记的公文语料 使用易混淆词表处理更多的错词    
    # f = open(vocabid_list_f, 'rb')
    # vocabid_list = pickle.load(f) 

    haidian_news('officialdoc/haidian_news_pseudo.txt', save_name, manual_change=True, vocabid_list=vocabid_list)