import jieba
import jieba.analyse
import os
import sys
import numpy as np
import math
import random
import chardet
import heapq
import pandas as pd
from matplotlib import pyplot as plt

class Weibo():
    def __init__(self):
        self.Lexicon = set([])
        self.final_word = []
        self.content = []
        self.time_lexicon = []
        self.time_word_list = []
        self.time_word_all = []
        for _ in range(9):
            self.time_lexicon.append([])
            self.time_word_list.append([])
            self.time_word_all.append([])
            self.final_word.append([])

    def stopword(self):
        stop_words=[]
        with open('stopwords.txt','r',encoding='UTF-8') as f:
            for line in f:
                stop_words.append(line.strip('\n'))
        # print(stop_words)
        return stop_words
    
    def word_cut(self, stop_words):
        f = open(file='content.txt',mode='rb')
        data = f.read()
        f.close()
        encode = chardet.detect(data)['encoding']
        with open('content.txt', 'r', encoding=encode) as f:
            for line in f:
                word_content = []
                line = line.strip('\n')
                part = jieba.analyse.extract_tags(line, allowPOS=('n','vn','v'))  # 只能从名词、人名、地名、动名词、动词中选取
                # part = jieba.analyse.extract_tags(line, allowPOS=('n','vn','v'))              # 只能从名词、动名词、动词中选取
                for word in part:
                    if word in stop_words or len(word)<2:
                        continue
                    elif word == ' ':
                        continue
                    else:
                        word_content.append(word)
                self.content.append(word_content)
                self.Lexicon = self.Lexicon | set(word_content)
        self.Lexicon = list(self.Lexicon)
        
        with open('time.txt', 'r', encoding=encode) as f:
            i = 0
            for line in f:
                line = line.strip('\n')
                line = line.split(' ')
                day = line[0].split('/')[-1]
                hour = line[1].split(':')[0]
                if int(day)==8:        # 3个小时为一个间隔，第一间隔
                    for h_i, h in enumerate(range(19, 24, 3)):
                        if int(hour)>=h and int(hour)<=h+2:
                            self.time_word_all[h_i].extend(self.content[i])
                            self.time_word_list[h_i].append(self.content[i])
                            break
                elif int(day)==9:
                    if int(hour)==0:
                        self.time_word_all[1].extend(self.content[i])
                        self.time_word_list[1].append(self.content[i])
                    else:
                        for h_i, h in enumerate(range(1, 20, 3)):
                            if int(hour)==h and int(hour)<=h+2:
                                self.time_word_all[h_i+2].extend(self.content[i])
                                self.time_word_list[h_i+2].append(self.content[i])
                                break
                i+=1
        for i in range(len(self.time_word_all)):
            for w in self.time_word_all[i]:
                if w not in self.time_lexicon[i]:
                    self.time_lexicon[i].append(w)

    def f_word(self):
        self.word_times = []
        for i in range(len(self.time_word_all)):
            word_times = np.zeros(len(self.time_lexicon[i]))
            for w in range(len(self.time_lexicon[i])):
                w_times = self.time_word_all[i].count(self.time_lexicon[i][w])
                word_times[w] = w_times
            self.word_times.append(list(word_times))

    def Wij(self, a, T):
        # print(self.time_lexicon)
        self.word_Wij = []
        for i in range(len(self.time_lexicon)):
            word_wij = np.zeros(len(self.time_lexicon[i]))
            if len(self.time_lexicon[i])==0:
                self.word_Wij.append([])
                continue
            Fmax = max(self.word_times[i])
            for tw in range(len(self.time_lexicon[i])):
                Fij = self.word_times[i][tw]
                Fiu = 0
                for u in range(i):
                    if self.word_times[i][tw] in self.time_lexicon[u]:
                        Fiu += self.word_times[u][self.time_lexicon[u].index(self.word_times[i][tw])]
                Gij = Fij * i / (Fiu+1)
                wij = math.log(Gij+1) + a*math.log((Fij/Fmax))
                word_wij[tw] = wij
            self.word_Wij.append(word_wij)
        for i in range(len(self.time_lexicon)):
            # for w in range(len(self.time_lexicon[i])):
                # if self.word_Wij[i][w]>=T:
            max_w = map(list(self.word_Wij[i]).index, heapq.nlargest(5, list(self.word_Wij[i])))
            max_w = list(max_w)
            for m in max_w:
                self.final_word[i].append(self.time_lexicon[i][m])      # 得到每个时间间隔中的词汇表
        for i in range(len(self.final_word)):
            self.final_word[i] = list(set(self.final_word[i]))

    def cluster(self):
        self.key_word = []
        for i in range(len(self.time_lexicon)):
            if len(self.time_lexicon[i])==0:
                self.key_word.append([])
                continue
            key_word = []
            for w in self.final_word[i]:
                dis_w = []
                if len(dis_w)==0:
                    key_word.append([w])
                    continue
                for k in range(len(key_word)):
                    P_max = -100000000	
                    for k_w in key_word[k]:
                        k_w_times = 0
                        w_times = 0
                        for info in self.time_word_list[i]:
                            if w in info and k_w in info:
                                k_w_times+=1
                                w_times+=1
                            elif w in info:
                                w_times+=1
                        P = k_w_times/w_times
                        if P>P_max:
                            P_max = P
                    if P_max==0:
                        dis_w.append(sys.maxsize)
                    else:
                        dis_w.append(1/P_max)
                min_dis = min(dis_w)
                if min_dis==sys.maxsize:
                    key_word.append([w])
                else:
                    min_dis_index = dis_w.index(min_dis)
                    key_word[min_dis_index].append(w)

            self.key_word.append(key_word)
        print(self.key_word)

weibo = Weibo()
stopwords = weibo.stopword()
weibo.word_cut(stopwords)
weibo.f_word()
weibo.Wij(1.25, 5)
weibo.cluster()
