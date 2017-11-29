import os

def get_id():
    f = open('/home/monkeys/workspace/SMP/src/1_BlogContent.txt')
    cur_dir = 1
    cur_sample = 0
    ids = []
    for i in range(1000000):
        row = f.readline()
        terms = row.split("")
        ids.append(terms[0])
    f.close()
    ids.sort()
    id_str = ""
    for id in ids:
        id_str += id+"\n"
    f = open("Corpus/ids.txt","w")
    f.write(id_str)
    f.close()

def save_corpus():
    f = open('/home/monkeys/workspace/SMP/src/1_BlogContent.txt')
    for i in range(1000000):
        row = f.readline()
        terms = row.split("")
        content = ""
        for term in terms:
            content += term+"\n"

        num = terms[0]
        num_int = int(num[1:])
        num = num_int/5000
        if num_int%5000==0:
            first = (num-1)*5000+1
            end = num*5000
        else:
            first = num*5000+1
            end = num*5000+5000
        str_first = str(first)
        str_end = str(end)
        for i in range(7-str_first.__len__()):
            str_first = "0" + str_first
        for i in range(7-str_end.__len__()):
            str_end = "0" + str_end
        dir_name = "D"+str_first+"-"+"D"+str_end
        print dir_name
        # print content
        f_blog = open("Corpus/"+dir_name+"/"+terms[0]+".txt","w")
        f_blog.write(content)
        f_blog.close()


# def save_corpus_2():


def creat_dir():
    first = 1
    end = 5000
    while(first<1000000):
        str_first = str(first)
        str_end = str(end)
        for i in range(7-str_first.__len__()):
            str_first = "0" + str_first
        for i in range(7-str_end.__len__()):
            str_end = "0" + str_end
        dir_name = "D"+str_first+"-"+"D"+str_end
        first += 5000
        end += 5000
        print dir_name
        os.makedirs("Corpus/"+dir_name)

def save_corpus_2():
    path = "/home/monkeys/Corpus"
    dirs = os.listdir(path)
    for dir in dirs:
        filenames = os.listdir(path+"/"+dir+"")
        for name in filenames:
            f = open(path+"/"+dir+"/"+name,"r")
            content = f.read()
            f.close()
            f = open("Corpus/"+name,"w")
            f.write(content)
            f.close()


def get_corpus_num():
    dirs = os.listdir("/Corpus")
    print len(dirs)
# get_id()
# creat_dir()

get_corpus_num()
