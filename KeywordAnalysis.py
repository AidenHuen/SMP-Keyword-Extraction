# coding=utf-8
import codecs
import os
import math
class KeywordAnalysis:

    def __init__(self):
        """
        构造函数
        """
        self.testset_path = "Data/TrainingTerms2/" # 测试集路径
        self.trainset_path = "Data/TrainingTerms2/"  # 训练集路径
        self.test_answer_path = "Data/TrainingSet.txt"  # 测试集结果路径
        self.idf_path = "Data/idf.txt"
        self.wordDict_path = "Data/wordDict2.txt"  # 已标注的词典
        self.stopword_path = "Data/Stopword.txt"  # 停用词路径
        self.tag_path = "Data/tag.txt"  # 特殊符号路径
        self.Symbol = ["。", "?", "？", "!", "！"]
        self.filenames = self.get_filenames(self.testset_path)
        self.training_filenames = self.get_filenames(self.trainset_path)
        self.Stopwords = self.get_stop_word()  # 读取停用词

        self.tags = self.load_tag()  # 读取待过滤符号集
        # for i in self.Stopwords:
        #     print i
        # self.weight = [0.4086, 0.3586, 0.3507, 0.2068, 0.0899, 0.0819, 0.0410, 0.0132, 0.0261, 0.0521, 0.0250, 0.0372]

       # 根据训练集训练的位置特征
        self.weight = [0.6391389432485323, 0.4042074363992172, 0.35861579414374443, 0.29405940594059404,
                       0.09207920792079208, 0.09465478841870824, 0.11206896551724138, 0.05387931034482758,
                       0.07247008973080758, 0.11839105417435675, 0.049397459559222666, 0.052891405094903025]

        self.idf_dict = self.load_idf() #　读取ｉｄｆ权值

        # 根据训练集训练的词性特征
        self.pos_weight = {'gp': 0.06832298136645963, 'gg': 0.0425531914893617,
                               'gc': 0.07142857142857142, 'gb': 0.016666666666666666,
                               'gm': 0.013333333333333334, 'gi': 0.15455128616332028,
                               'tg': 0.027777777777777776, 'dl': 0.008064516129032258,
                               'dg': 0.14285714285714285, 'usuo': 0.0022271714922048997,
                               'udeng': 0.0006060606060606061, 'ryt': 0.07692307692307693,
                               'd': 0.0002371996882518383, 'qt': 0.0005841121495327102,
                               'qv': 0.002976190476190476, 'h': 0.25, 'l': 0.0007062146892655367,
                               'p': 4.596221905593602e-05, 'pbei': 0.0012135922330097086,
                               't': 0.0004997501249375312, 'x': 0.016666666666666666,
                               'rr': 0.00011851149561507466, 'ry': 0.00101010101010101,
                               'rz': 0.00014843402107763098, 'r': 0.00042517006802721087,
                               'uzhe': 0.0053475935828877, 'uzhi': 0.0048543689320388345,
                               'pba': 0.0011494252873563218, 'bl': 0.005319148936170213,
                               'ude1': 2.4000384006144097e-05, 'ude2': 0.002207505518763797,
                               'ude3': 0.005681818181818182, 'ntu': 0.2, 'nmc': 0.14285714285714285,
                               'nto': 0.2, 'ntc': 0.22566371681415928, 'c': 8.007687379884689e-05,
                               'g': 1.0, 'k': 0.0018248175182481751, 'o': 0.1111111111111111,
                               's': 0.001366120218579235, 'w': 1.1377340887887682e-05,
                               'vshi': 0.00012130033964095099, 'nhd': 0.07142857142857142,
                               'cc': 8.610297916307904e-05, 'vyou': 0.0003637686431429611,
                               'nsf': 0.015151515151515152, 'vf': 0.0003046922608165753, 'ulian': 0.04,
                               'nrf': 0.07329842931937172, 'udh': 0.006172839506172839,
                               'b': 0.0001934984520123839, 'f': 0.00043959908563390187,
                               'uls': 0.0022675736961451248, 'j': 0.009433962264150943,
                               'n': 0.02694523120956441, 'rzv': 0.00030330603579011223,
                               'rzt': 0.013157894736842105, 'mq': 0.00024021138601969732,
                               'v': 4.5286436712204695e-05, 'z': 0.004347826086956522,
                               'ule': 0.00019157088122605365, 'vd': 0.0030120481927710845,
                               'ad': 0.0003502626970227671, 'ag': 0.006944444444444444,
                               'vg': 0.005952380952380952, 'vi': 0.00022517451024544022,
                               'nnd': 0.01282051282051282, 'vl': 0.001152073732718894,
                               'al': 0.008403361344537815, 'vn': 0.00011252391133115787,
                               'an': 0.0026109660574412533, 'vx': 0.047619047619047616,
                               'nnt': 0.02727272727272727, 'nf': 0.005494505494505495,
                               'ng': 0.0009319664492078285, 'nz': 0.08361284157524303,
                               'ryv': 0.000966183574879227, 'nr': 0.0196078431372549,
                               'ns': 0.002325581395348837, 'nt': 0.10344827586206896,
                               'rys': 0.014925373134328358, 'uyy': 0.003424657534246575,
                               'nis': 0.0009823182711198428, 'a': 6.840413160954922e-05,
                               'e': 0.030303030303030304, 'i': 0.01639344262295082,
                               'm': 0.0012547051442910915, 'q': 0.00020990764063811922,
                               'u': 0.006535947712418301, 'y': 0.0013513513513513514,
                               'rzs': 0.0015923566878980893, 'uguo': 0.0030959752321981426}
        for pos in self.pos_weight:
            if self.pos_weight[pos]<0.015:
                self.pos_weight[pos] = 0.015
            if 0.015 < self.pos_weight[pos] < 0.0385:
                self.pos_weight[pos] = 0.0385

        self.worst_tags = ["”", "“", "　","·","?", "本书", "项目第", "几个", "碎碎"]
        self.wordgroup_dict = self.get_wordgroup_dict()

    def weight_training(self):
        for i in range(len(self.weight)):
            for n in range(10):
                self.weight[i]+= n*0.001
                print self.weight
                print self.get_precision()

    def get_precision(self):
        """
        提取关键词，输出准确率
        :return:
        """
        self.get_more_stopword()
        result = self.analysis_all()
        for i in result:
            print i,
            for word in result[i]:
                print word,
            print
        test_set = self.load_test_answer_path()
        sum = 0.0
        num = 0
        for id in result.keys():
            # print id
            fact = result[id]
            test = test_set[id]
            for word in fact:
                sum += 1
                if word in test:
                    num += 1
        print "识别的词数：" + str(num)
        print "总的词数：" + str(sum)
        print "precision:" + str(num/sum)
        return num/sum

    def analysis(self,name):
        """
        对单篇ｂｌｏｇ进行分析
        :return:
        """
        word_dict = self.count_tf(name)  # 获取每个词的权重

        for word in word_dict:  #在先有权重中加入ｉｄｆ权值
            try:
                word_dict[word] *= self.idf_dict[word]
                # print "has",word
            except:
                word_dict[word] *= 13

        ranks = sorted(word_dict.iteritems(), key= lambda a:a[1], reverse=True)
        new_ranks = []
        ranks = ranks[:30]
        for i in range(len(ranks)):
            ranks[i] = list(ranks[i])
        for i in range(len(ranks)):
            if self.wordgroup_dict.has_key(ranks[i][0]) == True:
                ranks[i][0] = self.wordgroup_dict[ranks[i][0]]

        for term in ranks:
            n = 0
            judge = False
            for tag in self.worst_tags:
                if term[0].__contains__(tag) == True:
                    judge = True
                    break
            if judge == True:
                continue
            for tag in self.tags:
                if term[0].__contains__(tag) == True:
                    # print term[0], tag
                    n += 1
            if n < 2:#
                if  term[0].__contains__(" ") == False and len(term[0].decode("utf-8")) < 14:
                    new_ranks.append(term)
                elif term[0].__contains__(" ") == True:
                    new_ranks.append(term)
        ranks = []
        for term in new_ranks:
            if term[0].__contains__("-"):
                if len(term[0].decode("utf-8"))==len(term[0]):
                    ranks.append(term)
            else:
                ranks.append(term)
        # if name.replace(".txt","") == "D0434482":
        #     print ranks[0][0],ranks[1][0],ranks[2][0]
        return ranks


    def analysis_all(self):
        """
        对虽有ｂｌｏｇ进行主题词提取
        :return:
        """
        result = {}
        self.filenames = sorted(self.filenames)
        for name in self.filenames:

            ranks = self.analysis(name)
            result_words = []
            length = len(ranks)
            if length>3:
                length = 3
            for i in range(length):
                result_words.append(ranks[i][0])
            result[name.replace(".txt","")] = result_words
            print name.replace(".txt","")+","+",".join(result_words)
        return result

    def load_test_answer_path(self):
        """
        读取测试集的标注结果
        :return:
        """
        f = open(self.test_answer_path, "r")
        rows = f.readlines()
        f.close()
        test_dict = {}
        for row in rows:
            row = row.replace("\n","").replace("﻿","").replace("	","")
            terms = row.split("")
            test_dict[terms[0]] = terms[1:]
        # for test in test_dict:
        #     print test,test_dict[test]
        return test_dict

    def count_tf(self,name):
            """
            融合词性和位置特征，计算tf值
            :param name:
            :return:
            """
            title_words, sentence_words = self.get_word(name,self.testset_path)
            word_tag = self.get_word_pos_dict(name,self.testset_path)
            word_dict = {}
            for word in title_words:
                word_dict[word] = 0
            for words in sentence_words:
                for word in words:
                    word_dict[word] = 0

            if title_words.__len__()>0:
                try:
                    word_dict[title_words[0]] = self.weight[0]*self.pos_weight[word_tag[title_words[0]]]  # 标题首词权重
                except:
                    word_dict[title_words[0]] = self.weight[0] * 0.2
                try:
                    word_dict[title_words[title_words.__len__()-1]] = self.weight[1]*\
                        self.pos_weight[word_tag[title_words[title_words.__len__()-1]]]  # 标题未词权重
                except:
                    word_dict[title_words[title_words.__len__()-1]] = self.weight[1] * 0.15
                for i in range(1,title_words.__len__()-1):
                    try:
                        word_dict[title_words[i]] += self.weight[2]*self.pos_weight[word_tag[title_words[i]]]
                    except:
                        word_dict[title_words[i]] += self.weight[2] * 0.15
            for i in range(sentence_words.__len__()):
                if sentence_words[i].__len__()>1:
                    if i==0:
                        try:
                            word_dict[sentence_words[i][0]] += self.weight[3]*self.pos_weight[word_tag[sentence_words[i][0]]]
                        except:
                            word_dict[sentence_words[i][0]] += self.weight[3] * 0.1
                        try:
                            word_dict[sentence_words[i][len(sentence_words[i])-1]] += self.weight[4]*self.pos_weight[word_tag[sentence_words[i][len(sentence_words[i])-1]]]
                        except:
                            word_dict[sentence_words[i][len(sentence_words[i]) - 1]] += self.weight[4] * 0.1
                        for j in range(1,sentence_words[i].__len__()-1):
                            try:
                                word_dict[sentence_words[i][j]] += self.weight[5]*self.pos_weight[word_tag[sentence_words[i][j]]]
                            except:
                                word_dict[sentence_words[i][j]] += self.weight[5] * 0.1

                    elif i==sentence_words.__len__()-1:
                        try:
                            word_dict[sentence_words[i][0]] += self.weight[6]*self.pos_weight[word_tag[sentence_words[i][0]]]
                        except:
                            word_dict[sentence_words[i][0]] += self.weight[6] * 0.1
                        try:
                            word_dict[sentence_words[i][len(sentence_words[i])-1]] += self.weight[7]*self.pos_weight[word_tag[sentence_words[i][len(sentence_words[i])-1]]]
                        except:
                            word_dict[sentence_words[i][len(sentence_words[i]) - 1]] += self.weight[7] * 0.1

                        for j in range(1,sentence_words[i].__len__()-1):
                            try:
                                word_dict[sentence_words[i][j]] += self.weight[8]*self.pos_weight[word_tag[sentence_words[i][j]]]
                            except:
                                word_dict[sentence_words[i][j]] += self.weight[8] * 0.1
                    else:
                        try:
                            word_dict[sentence_words[i][0]] += self.weight[9]*self.pos_weight[word_tag[sentence_words[i][0]]]
                        except:
                            word_dict[sentence_words[i][0]] += self.weight[9] * 0.1
                        try:
                            word_dict[sentence_words[i][len(sentence_words[i])-1]] += self.weight[10]*self.pos_weight[word_tag[sentence_words[i][len(sentence_words[i])-1]]]
                        except:
                            word_dict[sentence_words[i][len(sentence_words[i]) - 1]] += self.weight[10] * 0.1
                        for j in range(1,sentence_words[i].__len__()-1):
                            try:
                                word_dict[sentence_words[i][j]] += self.weight[11]*self.pos_weight[word_tag[sentence_words[i][j]]]
                            except:
                                word_dict[sentence_words[i][j]] += self.weight[11] * 0.1
            return word_dict

    def training_position_feature(self):
        """
        读取训练集，并训练位置特征
        :return:
        """
        weight = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        num = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        test_set = self.load_test_answer_path()
        for name in self.training_filenames:
            title_words, sentence_words = self.get_word(name, self.trainset_path)
            correct_words = test_set[name.replace(".txt", "")]
            if title_words.__len__()>0:
                if title_words[0] in correct_words:
                    weight[0] += 1
                num[0] += 1
                if title_words[title_words.__len__()-1] in correct_words:
                    weight[1] += 1
                num[1] += 1
                for i in range(1,title_words.__len__()-1):
                    if title_words[i] in correct_words:
                        weight[2] += 1
                    num[2] += 1
            for i in range(sentence_words.__len__()):
                if sentence_words[i].__len__()>0:
                    if i==0:
                        if sentence_words[i][0] in correct_words:
                            weight[3] += 1
                        num[3] += 1
                        if sentence_words[i][len(sentence_words[i])-1] in correct_words:
                            weight[4] += 1
                        num[4] += 1
                        for j in range(1,sentence_words[i].__len__()-1):
                            if sentence_words[i][j] in correct_words:
                                weight[5] += 1
                            num[5] += 1

                    elif i==sentence_words.__len__()-1:
                        if sentence_words[i][0] in correct_words:
                            weight[6] += 1
                        num[6] += 1
                        if sentence_words[i][len(sentence_words[i])-1] in correct_words:
                            weight[7] += 1
                        num[7] += 1
                        for j in range(1,sentence_words[i].__len__()-1):
                            if sentence_words[i][j] in correct_words:
                                weight[8] += 1
                            num[8] += 1
                    else:
                        if sentence_words[i][0] in correct_words:
                            weight[9] += 1
                        num[9] += 1
                        if sentence_words[i][len(sentence_words[i])-1] in correct_words:
                            weight[10] += 1
                        num[10] += 1
                        for j in range(1,sentence_words[i].__len__()-1):
                            if sentence_words[i][j] in correct_words:
                                weight[11] += 1
                            num[11] += 1
        for i in range(weight.__len__()):
            weight[i] = weight[i]/float(num[i])
        print weight
            # print str(weight[i]/float(num[i]))+",",

    def get_stop_word(self):
        f = open(self.stopword_path,"r")
        rows = f.readlines()
        f.close()
        stop_word = {}
        for row in rows:
            row = row.decode("utf-8").encode("utf-8").replace("\n","")
            stop_word[row] =1
        return stop_word

    def get_filenames(self,name):
        """
        获取测试样本（已分词和词性标注)
        :return: 文档名
        """
        filenames = os.listdir(name)
        for i in range(filenames.__len__()):
            filenames[i] = filenames[i].replace("\n","")
        return filenames

    def get_terms(self, name,folder):
        """
        输入文件名及所在文件夹名，获取ｂｌｏｇ标题与正文的（词，词性）二元组集合
        :param name:
        :param folder:
        :return:
        """
        f=open(folder+name,"r")
        rows = f.readlines()
        for i in range(rows.__len__()):
            rows[i] = rows[i].lower()
        f.close()
        rows[0] = rows[0][1:]
        rows[0] = rows[0].decode("gbk",'ignore').encode("utf-8").replace("\n","").replace("　　","").replace(" ","").replace("		","")
        rows[1] = rows[1].decode("gbk",'ignore').encode("utf-8").replace("\n","").replace("　　","").replace(" ","").replace("		","")
        title_terms = rows[0].split("")
        content_terms = rows[1].split("")
        terms = []
        for term in title_terms:
                term = term.split("/")
                if term[0] != "" and term.__len__()==2:
                    terms.append(term)
                else:
                    if term[0] != "" and terms.__len__()>2:
                        new_term=[term[0]+"/"+term[1],term[2]]
                        terms.append(new_term)
                    else:
                        if term[0] != "":
                            print "can't get",term[0],term
        for term in content_terms:
                term = term.split("/")
                if term[0] != "" and term.__len__()==2:
                    terms.append(term)
                else:
                    if term[0] != "" and terms.__len__()>2:
                        new_term=[term[0]+"/"+term[1],term[2]]
                        terms.append(new_term)
                    else:
                        if term[0] != "":
                            print "can't get",term[0],term
        # for i in terms:
        #     print i[0],i[1]
        return terms

    def get_word(self, name, folder):
            """
            输入文件名及所在文件夹名，获取ｂｌｏｇ标题与正文的词集合
            :param name: 文件名
            :param folder: 所在文件夹名
            :return:
            """
            f=open(folder+name,"r")
            rows = f.readlines()
            f.close()
            rows[0] = rows[0][1:]

            rows[0] = rows[0].decode("gbk","replace").encode("utf-8").replace("\n","").replace("　　","").replace(" ","").replace("		","")
            rows[1] = rows[1].decode("gbk","replace").encode("utf-8").replace("\n","").replace("　　","").replace(" ","").replace("		","")


            # for tag in self.tags:
            #     rows[0] = rows[0].replace(tag, "")
            #     rows[1] = rows[1].replace(tag,"")
            title_terms = rows[0].split("")
            content_terms = rows[1].split("")
            title_words = []
            content_words = []
            for term in title_terms:
                term = term.split("/")
                if term[0] != "" and term.__len__() == 2:
                    title_words.append(term[0])
                else:
                    if term[0] != "" and term.__len__()>2:
                        new_term=[term[0]+"/"+term[1], term[2]]
                        title_words.append(new_term[0])
            for term in content_terms:
                term = term.split("/")
                if term[0] != "" and term.__len__() == 2:
                    content_words.append(term[0])
                else:
                    if term[0] != "" and term.__len__()>2:
                        new_term=[term[0]+"/"+term[1],term[2]]
                        content_words.append(new_term[0])
            title_words = self.filter_stopword(title_words)
            sentence_words = []
            sentence = []
            for i in range(len(content_words)):
                if content_words[i] not in self.Symbol:
                        sentence.append(content_words[i])
                else:
                    sentence_words.append(sentence)
                    sentence = []
                if i==content_words.__len__()-1 and content_words[i] not in self.Symbol:
                    sentence_words.append(sentence)
            for i in range(sentence_words.__len__()):
                sentence_words[i] = self.filter_stopword(sentence_words[i])
            # for sentence in sentence_words:
            #     for i in sentence:
            #         print i,
            #     print
            return title_words, sentence_words

    def filter_stopword(self,words):
        """
        过滤停用词
        :param words:
        :return:
        """
        new_words = []
        for word in words:
            # print word,
            if len(word.decode("utf-8"))>1: #　去单字词
                if not self.Stopwords.has_key(word):
                    # print word,len(word.decode("utf-8"))
                    new_words.append(word.lower())
        return new_words

    # def filter_stopword2(self,String):
    #     for word in self.sto

    def load_idf(self):
        """
        读取ｉｄｆ值
        :return:
        """
        f = open(self.idf_path,"r")
        rows = f.readlines()
        f.close()
        idf_dict = {}
        for row in rows:
            row = row.split(" ")
            # print row
            idf = float(row[row.__len__()-1])
            idf_dict[row[0]] = idf
        # for i in idf_dict:
        #     print i,idf_dict[i]
        return idf_dict

    def get_word_pos_dict(self,name,folder):
        terms = self.get_terms(name,folder)
        word_pos_dict = {}
        for term in terms:
                word_pos_dict[term[0]] = term[1]
        # for word in word_pos_dict:
        #     print word,word_pos_dict[word]
        return word_pos_dict

    def training_pos_feature(self):
        """
        训练词性特征
        :return:
        """
        blog_keywords = self.load_test_answer_path()
        pos_probability = {}
        pos_num = {}
        for name in self.training_filenames:
            terms = self.get_terms(name,self.trainset_path)
            # print terms
            for term in terms:
                try:
                    pos_num[term[1]] += 1
                except:
                    pos_num[term[1]] = 1
                # print term[0],term[1]
                if term[0] in blog_keywords[name.replace(".txt","")]:
                    # print term[0],term[1]
                    try:
                        pos_probability[term[1]] += 1.0
                    except:
                        pos_probability[term[1]] = 1.0
        for pos in pos_num:
            try:
                pos_probability[pos] = (pos_probability[pos]+1.0)/pos_num[pos]
            except:
                pos_probability[pos]  = 1.0/pos_num[pos]
        # for pos in pos_probability:
        #     pos_probability[pos] = math.log(math.e+math.e*pos_probability[pos])
        return pos_probability

    def load_tag(self):
        """
        读取待过滤的特殊标记
        :return:
        """
        tags = []
        f = open(self.tag_path,"r")
        rows = f.readlines()

        for row in rows:
            row = row.replace("\n","")
            if row != "":
                tags.append(row)
        return list(set(tags))

    def load_answer_freq(self):
        f = open(self.test_answer_path, "r")
        rows = f.readlines()
        f.close()
        answer_word_freq = {}
        for row in rows:
            row = row.replace("\n", "").replace("﻿", "").replace("	", "")
            terms = row.split("")
            for i in terms[1:]:
                if i != "":
                    try:
                        answer_word_freq[i] += 1
                    except:
                        answer_word_freq[i] = 1
        # for i in test_dict:
        #     print i,test_dict[i]
        return answer_word_freq

    def get_word_freq(self):
        word_freq = {}

        for name in self.filenames:
            ranks = self.analysis(name)
            for term in ranks[:15]:
                if term[0] != "":
                        try:
                            word_freq[term[0]] += 1
                        except:
                            word_freq[term[0]] = 1

        return word_freq

    def get_wordgroup_dict(self):
        f = open(self.wordDict_path)
        rows = f.readlines()
        word_group = []
        for row in rows:
            terms = row.lower().split(" ")
            if terms.__len__() > 2:
                word = ""
                for i in range(terms.__len__()-1):
                    if i == terms.__len__()-2:
                        word += terms[i]
                    else:
                        word += terms[i]+" "

                if word_group.__contains__(word) == False:
                    word_group.append(word)
        wordgroup_dict = {}
        for word in word_group:
            key = word.replace(" ","")
            wordgroup_dict[key] = word
        # for word in wordgroup_dict:
        #     print word,wordgroup_dict[word]
        return wordgroup_dict

    def get_training_keyword(self):
        f = open(self.wordDict_path)
        rows = f.readlines()
        keyword_dict = {}
        for row in rows:
            terms = row.split(" ")
            word = ""
            if terms.__len__() > 2:
                for i in range(terms.__len__()-1):
                    if i == terms.__len__()-2:
                        word += terms[i]
                    else:
                        word += terms[i]+" "
            else:
                word = terms[0]
            try:
                keyword_dict[word] += 1
            except:
                keyword_dict[word] = 1
        # for word in keyword_dict:
        #     print word,keyword_dict[word]
        return keyword_dict

    def get_more_stopword(self):
        answer_freq = self.load_answer_freq()
        word_freq = self.get_word_freq()
        key_word_freq = self.get_training_keyword()
        stop_words = {}
        for word in word_freq:
            if word not in answer_freq:
                if word not in key_word_freq:
                    stop_words[word] = 1
        # for word in answer_freq:
        #     print word
        # for word in stop_words:
        #     self.Stopwords[word] = 1
        #     print word

a = KeywordAnalysis()
a.get_precision()
