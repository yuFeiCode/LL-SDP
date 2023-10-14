import os

import pandas as pd
from tqdm import tqdm

from my_util import *

base_data_dir = '../datasets/preprocessed_data/'
base_original_data_dir = '../datasets/original/File-level/'

data_for_ngram_dir = '../datasets/n_gram_data/'
data_for_error_prone_dir = '../datasets/ErrorProne_data/'

proj_names = list(all_train_releases.keys())
############################################################################################
def export_df_to_files(data_df, code_file_dir, line_file_dir):
    '''
        input
            data_df(DataFrame): 
            code_file_dir: datastes/n_gram_data/文件名/src
            line_file_dir: datassets/n_gram_data/文件名/line_num

    '''
    #tqdm函数用于创建一个进度条，可以在循环中显示处理进度。
    for filename, df in tqdm(data_df.groupby('filename')):

        code_lines = list(df['code_line'])
        code_str = '\n'.join(code_lines)
        code_str = code_str.lower()  #所有的字符转换成小写形式
        line_num = list(df['line_number'])
        line_num = [str(l) for l in line_num]  #将数字变成字符

        code_filename = filename.replace('/','_').replace('.java','')+'.txt'
        line_filename = filename.replace('/','_').replace('.java','')+'_line_num.txt'
        #‘w’是指采取写模式打开文件，如果文件存在，则将其覆盖，如果不存在，则新建一个文件，
        #打开文件对象，并将其赋值给对象f，后面的f.write（）是对f进行的操作
        with open(code_file_dir+code_filename,'w') as f: #打开一个文件，并将字符串 code_str 写入该文件中。
            f.write(code_str)

        with open(line_file_dir+line_filename, 'w') as f:
            f.write('\n'.join(line_num))

def export_ngram_data_each_release(release, is_train = False):#默认为Flase，即导出非训练数据
    #存放处理过后的数据集的地方目录（n-gram）
    file_dir = data_for_ngram_dir+release+'/'
    file_src_dir = file_dir+'src/'
    file_line_num_dir = file_dir+'line_num/'
    #指定文件路径，存在的话就返回true
    if not os.path.exists(file_src_dir):
        os.makedirs(file_src_dir)

    if not os.path.exists(file_line_num_dir):
        os.makedirs(file_line_num_dir)
    #读取预处理之后的数据，并将其放在data_df的数据结构当中
    data_df = pd.read_csv(base_data_dir+release+'.csv', encoding='latin')

    # get clean files for training only
    if is_train:
        data_df = data_df[(data_df['is_test_file']==False) & (data_df['is_blank']==False) & (data_df['file-label']==False)]
    # get defective files for prediction only
    else:
        data_df = data_df[(data_df['is_test_file']==False) & (data_df['is_blank']==False) & (data_df['file-label']==True)]

    data_df = data_df.fillna('')
    
    export_df_to_files(data_df, file_src_dir, file_line_num_dir)

def export_data_all_releases(proj_name):
    train_rel = all_train_releases[proj_name]
    eval_rels = all_eval_releases[proj_name]

    export_ngram_data_each_release(train_rel, True)
    #分界点，也就是上面的是train文件，下面的是测试文件
    for rel in eval_rels:
        export_ngram_data_each_release(rel, False)
        # break

def export_ngram_data_all_projs():
    for proj in proj_names:
        export_data_all_releases(proj)
        print('finish',proj)
###############################################################################
def export_errorprone_data(proj_name):
    cur_eval_rels = all_eval_releases[proj_name][1:]

    for rel in cur_eval_rels:

        save_dir = data_for_error_prone_dir+rel+'/'

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        #从activemq中的5.2.0到5.8.0一直进行
        data_df = pd.read_csv(base_original_data_dir+rel+'_ground-truth-files_dataset.csv', encoding='latin')

        data_df = data_df[data_df['Bug']==True] #这行代码的作用是从数据框 data_df 中仅保留 'Bug' 列值为 True 的行，删除其他行

        for filename, df in data_df.groupby('File'): #File means FileName

            if 'test' in filename or '.java' not in filename:
                continue

            filename = filename.replace('/','_')
            #因为按照File分组之后，由于File是唯一的，所以每个分组里面只有一行，那么list(df['SRC'])[0]
            #其实就相当于提取SRC
            code = list(df['SRC'])[0].strip()

            with open(save_dir+filename,'w') as f:
                f.write(code)

        print('finish release',rel)


def export_error_prone_data_all_projs():
    for proj in proj_names:
        export_errorprone_data(proj)
        print('finish',proj)
##############################################################################
export_ngram_data_all_projs()
export_error_prone_data_all_projs()