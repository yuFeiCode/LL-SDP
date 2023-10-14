import os, sys

from gensim.models import Word2Vec

import more_itertools

from DeepLineDP_model import *
from my_util import *


def train_word2vec_model(dataset_name, embedding_dim = 50):

    w2v_path = get_w2v_path()

    save_path = w2v_path+'/'+dataset_name+'-'+str(embedding_dim)+'dim.bin'

    if os.path.exists(save_path):
        print('word2vec model at {} is already exists'.format(save_path))
        return

    if not os.path.exists(w2v_path):
        os.makedirs(w2v_path)

    train_rel = all_train_releases[dataset_name]#train_rel == activemq 5.0.0

    train_df = get_df(train_rel)  #返回一个df对象，包含的是preprocessed_data的非空白和非测试行
    #每个分组之后的每个文件的三维列表，文件名->行名->单词，还会返回每个文件到底是bug还是不是bug
    train_code_3d, _ = get_code3d_and_label(train_df, True)

    all_texts = list(more_itertools.collapse(train_code_3d[:],levels=1))

    word2vec = Word2Vec(all_texts,vector_size=embedding_dim, min_count=1,sorted_vocab=1)

    word2vec.save(save_path)
    print('save word2vec model at path {} done'.format(save_path))

#这行代码是获取命令行参数中的第一个参数，并将其赋值给变量 p。
#例如，如果在命令行中执行脚本 python script.py argument1，
#那么 sys.argv[1] 将获取到字符串 "argument1"，然后赋值给变量 p。
#这样，变量 p 就可以在脚本中使用，代表命令行中传入的第一个参数的值
p = sys.argv[1]

train_word2vec_model(p,50)