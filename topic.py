import jieba
import jieba.analyse
import os
import numpy as np
import math
import random
import chardet
import heapq
import pandas as pd
import argparse
from matplotlib import pyplot as plt

class Topic_Detecton():
    def __init__(self, train_path, cluster, max_iter, threshold, max_tf_idf=0):
        self.train_path = train_path
        self.cluster = cluster
        self.max_iter = max_iter
        self.distance_threshold = threshold
        self.max_tf_idf = max_tf_idf

        self.train_file_num = 0
        self.vector_length = 0
        self.Lexicon = set([])
        self.file_word = []
        self.test_file_word = []
        self.word_tf_idf = []
        self.center_all = []
        self.final_cluster = []
        self.file_name = []
        self.file_test_name = []
        
        self.SSE_total = {}
        self.profile_total = {}

    def stopword(self):
        stop_words=[]
        with open('stopwords.txt','r',encoding='UTF-8') as f:
            for line in f:
                stop_words.append(line.strip('\n'))
        # print(stop_words)
        return stop_words

    def word_cut(self, txt_path, stop_words, test=False):
        word_content = []
        f = open(file=txt_path,mode='rb')
        data = f.read()
        f.close()
        encode = chardet.detect(data)['encoding']
        with open(txt_path, 'r', encoding=encode) as f:
            for line in f:
                line = line.strip('\n')
                # part = jieba.analyse.extract_tags(line, allowPOS=('n','nr','ns','vn','v'))  # 只能从名词、人名、地名、动名词、动词中选取
                part = jieba.analyse.extract_tags(line, allowPOS=('n','vn','v'))              # 只能从名词、动名词、动词中选取
                for word in part:
                    if word in stop_words or len(word)<2:
                        continue
                    elif word == ' ':
                        continue
                    else:
                        word_content.append(word)
        if test:
            self.test_file_word.append(word_content)
            return word_content
        else:
            self.file_word.append(word_content)
            return word_content

    def get_file(self, file_path, test=False):
        if not file_path:
            file_path = self.train_path
        stop_words = self.stopword()
        for root, dirs, _ in os.walk(file_path):
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                files = os.listdir(dir_path)
                for file in files:
                    print(file + " 开始处理----------")
                    file_path = os.path.join(dir_path, file)
                    if not test:
                        word_content = self.word_cut(file_path, stop_words)
                        self.Lexicon = self.Lexicon | set(word_content)
                        print("------------处理完毕----------")
                        self.train_file_num += 1
                        self.file_name.append([file, -1])
                    else:
                        word_content = self.word_cut(file_path, stop_words, True)
                        self.file_test_name.append([file, -1])
        if not test:
            self.Lexicon = list(self.Lexicon)
        print("词库搭建完毕")

    def IDF(self):
        print("开始计算IDF")
        self.Lexicon_idf = np.ones(len(self.Lexicon))
        for i in range(0, len(self.Lexicon)):
            word = self.Lexicon[i]
            for content in self.file_word:
                if word in content:
                    self.Lexicon_idf[i] += 1
            self.Lexicon_idf[i] = self.train_file_num / self.Lexicon_idf[i]
            self.Lexicon_idf[i] = math.log(self.Lexicon_idf[i]) 
        print("IDF计算完毕")

    def TF(self, word_content):
        tf = np.zeros(len(self.Lexicon))
        for i in range(0, len(word_content)):
            word = word_content[i]
            word_index = self.Lexicon.index(word)
            tf[word_index] += 1
            self.Lexicon_idf[word_index] += 1
        tf = tf / len(word_content)
        return tf

    def TF_IDF(self):
        self.IDF()
        all = len(self.file_word)
        handle = 0
        word_tf_idf_init = []
        for word_content in self.file_word:
            tf = self.TF(word_content)
            tf_idf = tf * self.Lexicon_idf
            # 不等于0表示要对tf_idf进行过滤，只选取前x个，那么其余的归零
            if self.max_tf_idf!=0:
                max = map(list(tf_idf).index, heapq.nlargest(self.max_tf_idf, list(tf_idf)))
                max = list(max)
                for i in range(0,len(tf_idf)):
                    if i not in max:
                        tf_idf[i] = 0
            word_tf_idf_init.append(tf_idf)
            handle += 1
            print("已处理 "+str(handle)+" 剩余："+str(all-handle))
            
        word_tf_idf_init = np.array(word_tf_idf_init)
        self.word_tf_idf = word_tf_idf_init
        self.vector_length = len(self.word_tf_idf[0])

    def cos_calculate(self, list1, list2):
        numerator = float(list1.dot(list2.T))
        denominator = np.linalg.norm(list1) * np.linalg.norm(list2)
        distance = 0.5 + 0.5*(numerator/denominator)
        return distance

    def Euclid_calculate(self, list1, list2):
        distance = np.sum(np.power(list1-list2, 2))
        distance = math.sqrt(distance)
        return distance

    def sse_calculate(self, list1, list2):
        distance = np.sum(np.power(list1-list2, 2))
        return distance

    def K_Means_foundation(self, center):
        pre_sse = 0
        for i in range(0,self.max_iter):

            print("开始第 "+str(i+1)+" 次迭代")
            cluster_result = []
            for _ in range(0, self.cluster):
                cluster_result.append([])
            # 对于每一个文本，寻找最近的质心并放入最终簇
            for con_i in range(0,len(self.word_tf_idf)):
                content = self.word_tf_idf[con_i]
                max_cos = -2
                cos_index = 0
                for c_i in range(0,len(center)):
                    cos = self.cos_calculate(content, center[c_i])
                    if cos > max_cos:
                        max_cos = cos
                        cos_index = c_i
                cluster_result[cos_index].append(content)
                self.file_name[con_i][1] = cos_index                # 更新簇ID
            # print(self.file_name)
            new_center = []
            sse = self.SSE(cluster_result, center, self.turn)
            profile = self.Profile_Analysis(cluster_result, center, self.turn)
            if math.fabs(sse-pre_sse)<self.distance_threshold:
                self.final_cluster = cluster_result
                self.center_all.append(center)
                # self.final_center = center
                break
            else:
                pre_sse = sse
                for c_i in range(0,len(cluster_result)):
                    new_c = []
                    new_c = np.mean(cluster_result[c_i], axis=0)              # 求平均值（新的聚类中心）
                    new_center.append(new_c)
                center.clear()
                center = new_center

    def K_Means(self):
        init_index = random.sample(range(0,len(self.word_tf_idf)-1), self.cluster)
        center = []
        for i in init_index:
            center.append(self.word_tf_idf[i])
            self.file_name[i][1] = i
        self.K_Means_foundation(center)

    def closet_distance(self, word_vector, center):
        max_cos = -2
        for _, c in enumerate(center):
            dis = self.cos_calculate(c, word_vector)
            max_cos = max(dis, max_cos)
        return max_cos

    def K_Means_pp(self):
        init_index = random.randint(1, len(self.word_tf_idf)-1)
        center = []
        center.append(self.word_tf_idf[init_index])
        self.file_name[init_index][1] = 0
        distance = [0.0 for _ in range(len(self.word_tf_idf))]
        for c in range(1, self.cluster):
            total = 0.0
            for i, word_vector in enumerate(self.word_tf_idf):
                d = (1-self.closet_distance(word_vector, center))
                total += d
                distance[i] = d
            total *= random.random()                                    # 乘以一个0-1的随机数
            for i, d in enumerate(distance):                            # 判断落于哪一个区域
                total -= d
                if total > 0:                   
                    continue                
                center.append(self.word_tf_idf[i])                      # 将选中的点放入聚类中心的列表中
                self.file_name[i][1] = c+1                              # 更新该向量对应的文本簇名
                break
        self.K_Means_foundation(center)

    def SSE(self, cluster, center, times):
        sse = 0
        for i in range(self.cluster):
            sse += self.sse_calculate(cluster[i], center[i])
        sse = sse / self.cluster
        self.SSE_total[times].append(sse)
        return sse

    def Profile_Analysis(self, cluster, center, times):
        for c in range(self.cluster):
            profile_dist = 0.0
            for word in cluster[c]:
                dist_in = 0.0
                dist_out = []
                for else_c in range(self.cluster):
                    if c==else_c:           # 簇内其他点的距离
                        for else_cc in cluster[else_c]:
                            dist_in += self.Euclid_calculate(else_cc, word)
                    else:
                        dist_out_ = 0.0
                        for else_cc in cluster[else_c]:
                            dist_out_ += self.Euclid_calculate(else_cc, word)
                        if len(cluster[else_c])==0:
                            dist_out.append(float("inf"))
                        else:
                            dist_out.append(dist_out_/len(cluster[else_c]))
                if len(cluster[c])==1:
                    dist_in = 0
                else:
                    dist_in = dist_in / (len(cluster[c])-1)
                dist_out = min(dist_out)
                dist_max = max(dist_in, dist_out)
                S = (dist_out - dist_in)/dist_max
                profile_dist += S
            profile_dist = profile_dist / (len(cluster[c]))
            # profile_cluster.append(profile_dist)
            self.profile_total[times][c].append(profile_dist)
        # self.profile_total[times].append(profile_cluster)
        return profile_dist
                    
    def write_cluster(self, writer, sheet_name='test1'):
        write = []
        for _ in range(0,self.cluster):
            write.append([])
        for f in self.file_name:
            # print("文件："+f[0]+"的聚类结果为："+str(f[1]))
            write[f[1]].append(f[0])
        message = {
            "第一簇":write[0],
            "第二簇":write[1],
            "第三簇":write[2],
            "第四簇":write[3],
            "第五簇":write[4],
            "第六簇":write[5]
        }
        message = pd.DataFrame.from_dict(message, orient='index')
        message = message.transpose()
        message.to_excel(writer, sheet_name=sheet_name)
        print("文件写入完毕")

    def draw_sse(self, list_):
        x = [j+1 for j in range(len(list_))]
        y = list_
        plt.plot(x, y)
        plt.xlabel('迭代次数')
        plt.ylabel('SSE')
        plt.title('第 '+str(self.turn)+' 次测试')
        plt.savefig(str(self.turn)+'_SSE.png')
        plt.clf()

    def draw_profile(self, list_):
        x = [j+1 for j in range(len(list_[0]))]
        i = 1
        for l in list_:
            y = l
            label = 'cluster'+str(i)
            plt.plot(x, y, label=label)
            i+=1
        plt.xlabel('迭代次数')
        plt.ylabel('profile')
        plt.title('第 '+str(self.turn)+' 次测试')
        plt.legend()
        plt.savefig(str(self.turn)+'_profile.png')
        plt.clf()

    def show_topic(self, center_list, title):
        i=1
        fig = plt.figure(figsize=(16,12))
        for center in center_list:
            plt.subplot(3,2,i)
            max = map(list(center).index, heapq.nlargest(10, list(center)))
            max = list(max)
            x_name = [self.Lexicon[m] for m in max]
            y = [center[m] for m in max]
            x = range(len(x_name))
            plt.bar(x, y)
            plt.xticks(x, x_name, size = 17)
            plt.title('cluster'+str(i))
            i+=1
        fig.tight_layout()
        plt.savefig(str(title)+'_test.png')
        plt.clf()


    def test(self, times):
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus']=False

        writer = pd.ExcelWriter(r'簇结果.xlsx') # pylint: disable=abstract-class-instantiated
        self.get_file('')
        self.TF_IDF()

        min_sse = float("inf")
        min_sse_index = 0
        min_sse_list = []
        for i in range(times):
            self.turn = i+1
            self.SSE_total.setdefault(i+1,[])
            self.profile_total.setdefault(i+1,[])
            for _ in range(self.cluster):
                self.profile_total[i+1].append([])
            self.K_Means_pp()
            print(self.SSE_total[i+1])
            if self.SSE_total[i+1][-1]<min_sse:
                min_sse = self.SSE_total[i+1][-1]
                min_sse_index = i
            min_sse_list.append((self.turn, min_sse))
            self.write_cluster(writer, 'test'+str(i))
            self.draw_sse(self.SSE_total[self.turn])
            self.draw_profile(self.profile_total[self.turn])
            self.show_topic(self.center_all[-1], self.turn)

        writer.save()
        writer.close()
        # self.show_topic(self.center_all[min_sse_index])
        print("SSE如下所示：")
        print(min_sse_list)

    def predict(self, filepath):
        center_index = input("请输入选择的测试结果")
        category = []
        print("请按照结果依次输入簇1-6的类别名，如C1")
        for i in range(6):
            category.append(input())
        center = self.center_all[int(center_index)]
        self.get_file('test', True)

        all = len(self.test_file_word)
        right = 0
        for t in range(len(self.test_file_word)):
            test_tf_idf = np.zeros(len(self.Lexicon))
            for w in range(len(self.Lexicon)):
                tf = self.test_file_word[t].count(self.Lexicon[w])
                idf = self.Lexicon_idf[w]
                test_tf_idf[w] = tf * idf
            max_cos = -2
            max_index = -1
            for c in range(len(center)):
                cos = self.cos_calculate(test_tf_idf, center[c])
                if cos>max_cos:
                    max_cos = cos
                    max_index = c
            self.file_test_name[t][1] = max_index
            cate = self.file_test_name[t][0].split('-')[0]
            if cate == category[max_index]:
                print("类别为："+cate)
                print("归类正确")
                right += 1
            else:
                print("归类错误")
                print("该类为："+cate+" 应在簇："+str(category.index(cate))+" 但是其在簇："+str(max_index))
        right = round((right/all),2)
        print("正确率为："+str(right))

        
parser = argparse.ArgumentParser()
parser.add_argument("--train_path",type=str,default='train30',help="训练数据集，内含多个子文件夹")
parser.add_argument("--test_path",type=str,default='test',help="测试数据集，内含多个文件")
parser.add_argument("--K",type=int,default=6,help="K值")
parser.add_argument("--iter",type=int,default=100,help="最大迭代次数")
parser.add_argument("--threshold",type=float,default=0.0005,help="停止条件")
parser.add_argument("--times",type=int,default=1,help="测试次数")
args = parser.parse_args()

detection = Topic_Detecton(args.train_path, args.K, args.iter, args.threshold)
detection.test(args.times)
detection.predict(args.test_path)


# detection.get_file()
# detection.TF_IDF()
# detection.K_Means_pp()
# detection.SSE()
# detection.write_cluster()
# stop_words = detection.stopword()
# detection.word_cut('C4-Literature01.txt', stop_words)


# plt.rcParams['font.family'] = ['sans-serif']
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus']=False
# detection.get_file()
# detection.TF_IDF()
# index = detection.file_name.index(['C4-Literature47.txt',-1])
# detection.show_topic([detection.word_tf_idf[index]], 'C4-Literature47')

# index = detection.file_name.index(['C5-Education026.txt',-1])
# detection.show_topic([detection.word_tf_idf[index]], 'C5-Education026')

# index = detection.file_name.index(['C7-History018.txt',-1])
# detection.show_topic([detection.word_tf_idf[index]], 'C7-History018')

# index = detection.file_name.index(['C39-Sports0005.txt',-1])
# detection.show_topic([detection.word_tf_idf[index]], 'C39-Sports0005')