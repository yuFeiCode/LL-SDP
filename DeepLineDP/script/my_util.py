import re

import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

max_seq_len = 50
#训练集
all_train_releases = {'activemq': 'activemq-5.0.0', 'camel': 'camel-1.4.0', 'derby': 'derby-10.2.1.6', 
                      'groovy': 'groovy-1_5_7', 'hbase': 'hbase-0.94.0', 'hive': 'hive-0.9.0', 
                      'jruby': 'jruby-1.1', 'lucene': 'lucene-2.3.0', 'wicket': 'wicket-1.3.0-incubating-beta-1'}
#测试集
all_eval_releases = {'activemq': ['activemq-5.1.0', 'activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0'], 
                     'camel': ['camel-2.9.0', 'camel-2.10.0', 'camel-2.11.0'], 
                     'derby': ['derby-10.3.1.4', 'derby-10.5.1.1'], 
                     'groovy': ['groovy-1_6_BETA_1', 'groovy-1_6_BETA_2'], 
                     'hbase': ['hbase-0.95.0', 'hbase-0.95.2'], 'hive': ['hive-0.10.0', 'hive-0.12.0'], 
                     'jruby': ['jruby-1.4.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1'], 
                     'lucene': ['lucene-2.9.0', 'lucene-3.0.0', 'lucene-3.1'], 
                     'wicket': ['wicket-1.3.0-beta2', 'wicket-1.5.3']}



all_releases = {'activemq': ['activemq-5.0.0', 'activemq-5.1.0', 'activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0'], 
                     'camel': ['camel-1.4.0', 'camel-2.9.0', 'camel-2.10.0', 'camel-2.11.0'], 
                     'derby': ['derby-10.2.1.6', 'derby-10.3.1.4', 'derby-10.5.1.1'], 
                     'groovy': ['groovy-1_5_7', 'groovy-1_6_BETA_1', 'groovy-1_6_BETA_2'], 
                     'hbase': ['hbase-0.94.0', 'hbase-0.95.0', 'hbase-0.95.2'], 'hive': ['hive-0.9.0', 'hive-0.10.0', 'hive-0.12.0'], 
                     'jruby': ['jruby-1.1', 'jruby-1.4.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1'], 
                     'lucene': ['lucene-2.3.0', 'lucene-2.9.0', 'lucene-3.0.0', 'lucene-3.1'], 
                     'wicket': ['wicket-1.3.0-incubating-beta-1', 'wicket-1.3.0-beta2', 'wicket-1.5.3']}

all_projs = list(all_train_releases.keys())

file_lvl_gt = '../datasets/preprocessed_data/'


word2vec_dir = '../output/Word2Vec_model/' 

def get_df(rel, is_baseline=False): #默认是Flase

    if is_baseline:
        df = pd.read_csv('../'+file_lvl_gt+rel+".csv")

    else:
        df = pd.read_csv(file_lvl_gt+rel+".csv")

    df = df.fillna('')

    df = df[df['is_blank']==False]
    df = df[df['is_test_file']==False]

    return df

def prepare_code2d(code_list, to_lowercase = False):
    '''
        input
            code_list (list): list that contains code each line (in str format)
        output
            code2d (nested list): a list that contains list of tokens with padding by '<pad>'
    '''
    code2d = []

    for c in code_list:
        #将c中的连续空白字符换成单个空格
        #'\s' 是一个特殊字符类，用于匹配任意空白字符。而 '+' 是一个量词，表示匹配前面的模式一次或多次。
        c = re.sub('\\s+',' ',c)

        if to_lowercase:
            c = c.lower()
        #对变量 c 进行处理，去除首尾空白字符，并将其拆分为一个标记列表。
        #这个操作可以用于将字符串按照空格进行分隔，以便进一步对每个标记进行处理或分析。
        token_list = c.strip().split()
        #这样一行的代码就会被分成一个个的tokens
        total_tokens = len(token_list)
        
        token_list = token_list[:max_seq_len] #embeding_size == 50
        #如果长度不够50的话，就用pad俩不起剩余的
        if total_tokens < max_seq_len:
            token_list = token_list + ['<pad>']*(max_seq_len-total_tokens)

        code2d.append(token_list)

    return code2d #code2d是一个二维的列表
    
def get_code3d_and_label(df, to_lowercase = False):
    '''
        input
            df (DataFrame): a dataframe from get_df()
        output
            #嵌套列表
            code3d (nested list): a list of code2d from prepare_code2d()
            all_file_label (list): a list of file-level label
    '''

    code3d = []
    all_file_label = []

    for filename, group_df in df.groupby('filename'):
        #原来的file_label是字符型的，现在将其转换成bool型
        file_label = bool(group_df['file-label'].unique())
        #将所有的代码行转换成一个列表
        code = list(group_df['code_line'])

        code2d = prepare_code2d(code, to_lowercase) #此时的to_lowercase是True
        code3d.append(code2d)

        all_file_label.append(file_label)

    return code3d, all_file_label

def get_w2v_weight_for_deep_learning_models(word2vec_model, embed_dim):
    #将Word2Vec模型中的词向量矩阵转换为PyTorch的FloatTensor类型，并将其移动到GPU上
    #word2vec_model.wv.vectors是Word2Vec模型中训练得到的词向量矩阵，它的形状为(词汇量大小, 词嵌入维度)。
    word2vec_weights = torch.FloatTensor(word2vec_model.wv.vectors).cuda()
    
    # add zero vector for unknown tokens
    # concatnate
    word2vec_weights = torch.cat((word2vec_weights, torch.zeros(1,embed_dim).cuda()))

    return word2vec_weights

def pad_code(code_list_3d,max_sent_len,limit_sent_len=True, mode='train'):
    paded = []
    
    for file in code_list_3d:
        sent_list = []
        #是对句子长度进行处理
        #句子过长的话，就直接取前面的max
        for line in file:
            new_line = line
            if len(line) > max_seq_len:
                new_line = line[:max_seq_len]
            sent_list.append(new_line)
        #是对句子数量进行处理
        #一个文件里面句子如果不到max的话，需要补空白行
        if mode == 'train':
            if max_sent_len-len(file) > 0:
                for i in range(0,max_sent_len-len(file)):
                    sent_list.append([0]*max_seq_len)

        if limit_sent_len:    
            paded.append(sent_list[:max_sent_len])
        else:
            paded.append(sent_list)
        
    return paded

def get_dataloader(code_vec, label_list,batch_size, max_sent_len):
    y_tensor =  torch.cuda.FloatTensor([label for label in label_list])
    #对输入的3D文本序列进行填充，以确保每个文件内的句子具有相同的长度，且每个文件具有相同数量的句子
    code_vec_pad = pad_code(code_vec,max_sent_len)
    #将填充后的文本序列 code_vec_pad 和标签张量 y_tensor 组合成一个数据集。
    tensor_dataset = TensorDataset(torch.tensor(code_vec_pad), y_tensor)
    #这部分代码创建了一个数据加载器 dl，将数据集 tensor_dataset 包装在其中。
    #shuffle=True表示在每个训练轮次之前对数据进行随机洗牌，
    #batch_size指定了每个批次的大小，
    #drop_last=True表示如果数据集大小不能被批次大小整除，最后一个不完整的批次将被丢弃
    dl = DataLoader(tensor_dataset,shuffle=True,batch_size=batch_size,drop_last=True)
    
    return dl

def get_x_vec(code_3d, word2vec):
    x_vec = [[[word2vec.wv.key_to_index[token] if token in word2vec.wv.key_to_index else len(word2vec.wv.key_to_index) for token in text]
         for text in texts] for texts in code_3d]
    
    return x_vec