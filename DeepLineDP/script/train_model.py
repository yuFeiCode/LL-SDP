import os, re, argparse

import torch.optim as optim

import numpy as np
import pandas as pd

from gensim.models import Word2Vec

from tqdm import tqdm

from sklearn.utils import compute_class_weight

from DeepLineDP_model import *
from my_util import *
#设置pytorch的随机种子为0，以便于实验结果可以复现
torch.manual_seed(0)
#创建一个参数解析器对象 arg，用于解析命令行传递的参数和选项。
arg = argparse.ArgumentParser()

arg.add_argument('-dataset',type=str, default='activemq', help='software project name (lowercase)')
arg.add_argument('-batch_size', type=int, default=32) #单次传递给程序用以训练的数据（样本）个数 https://blog.csdn.net/u011699626/article/details/120352398
arg.add_argument('-num_epochs', type=int, default=10) #每个训练集训练的次数
arg.add_argument('-embed_dim', type=int, default=50, help='word embedding size') #深度学习词向量的维度
arg.add_argument('-word_gru_hidden_dim', type=int, default=64, help='word attention hidden size')          
arg.add_argument('-sent_gru_hidden_dim', type=int, default=64, help='sentence attention hidden size')      
arg.add_argument('-word_gru_num_layers', type=int, default=1, help='number of GRU layer at word level')
arg.add_argument('-sent_gru_num_layers', type=int, default=1, help='number of GRU layer at sentence level')
arg.add_argument('-dropout', type=float, default=0.2, help='dropout rate')  #防止过拟合
arg.add_argument('-lr', type=float, default=0.001, help='learning rate') #学习率 用于梯度下降的步长
arg.add_argument('-exp_name',type=str,default='') #实验名称 或者 模型名称
#将命令行中的参数值转换为相应的数据类型，并将其存储在命名空间对象 args 中
#我们就可以通过 args 对象的属性来访问和使用命令行参数的值
args = arg.parse_args()

# model setting
batch_size = args.batch_size
num_epochs = args.num_epochs
max_grad_norm = 5 #防止梯度爆炸 计算梯度向量的范数 并且与max——grad比较
embed_dim = args.embed_dim
word_gru_hidden_dim = args.word_gru_hidden_dim #单词级别的GRU隐藏单元的维度
sent_gru_hidden_dim = args.sent_gru_hidden_dim #句子级别的GRU隐藏单元的维度
word_gru_num_layers = args.word_gru_num_layers #单词级别GRU的层数
sent_gru_num_layers = args.sent_gru_num_layers #句子级别GRU的层数
word_att_dim = 64 #单词级别注意力机制的维度
sent_att_dim = 64 #句子级别注意力机制的维度
use_layer_norm = True  #使用层归化一，层归一化是一种正则化技术，可以提高网络的训练效果。
dropout = args.dropout
lr = args.lr

save_every_epochs = 1 #模型每隔多少个epochs保存一次
exp_name = args.exp_name
#???????????????????????????????????????????????????????????????????????????????????
max_train_LOC = 900 #用于限制训练数据的最大长度。 long of coverage

prediction_dir = '../output/prediction/DeepLineDP/' # 和当前的代码无关
save_model_dir = '../output/model/DeepLineDP/' #训练完的数据存放地址

file_lvl_gt = '../datasets/preprocessed_data/' #预处理的数据

weight_dict = {} #字典类型

def get_loss_weight(labels):
    '''numpy 数组 但是在 torch 中我们叫做 tensor 张量
       input
            labels: a PyTorch tensor that contains labels
        output
            weight_tensor: a PyTorch tensor that contains weight of defect/clean class
    '''
    #cpu() 方法用于将张量从 GPU 内存中移动到 CPU 内存中。这是因为后续的操作需要在 CPU 上进行，而不是在 GPU 上。
    #numpy() 方法将 PyTorch 张量转换为 NumPy 数组。
    #squeeze() 方法用于去除张量中维度为1的维度。如果张量的某个维度大小为1，那么该维度可以被去除，
    #从而减少维度的数量。这样可以简化后续的操作。
    #tolist() 方法将 NumPy 数组转换为 Python 列表
    label_list = labels.cpu().numpy().squeeze().tolist()

    weight_list = []

    for lab in label_list:
        if lab == 0:
            weight_list.append(weight_dict['clean'])
        else:
            weight_list.append(weight_dict['defect'])
    #将 Python 列表 weight_list 转换为一个形状为 (-1, 1) 的二维 PyTorch 张量，并将该张量移动到 GPU 上进行计算。
    weight_tensor = torch.tensor(weight_list).reshape(-1,1).cuda()
    return weight_tensor

def train_model(dataset_name):

    loss_dir = '../output/loss/DeepLineDP/'
    actual_save_model_dir = save_model_dir+dataset_name+'/'

    if not exp_name == '':
        actual_save_model_dir = actual_save_model_dir+exp_name+'/'
        loss_dir = loss_dir + exp_name

    if not os.path.exists(actual_save_model_dir):
        os.makedirs(actual_save_model_dir)

    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)

    train_rel = all_train_releases[dataset_name] #训练集
    valid_rel = all_eval_releases[dataset_name][0] #测试集

    train_df = get_df(train_rel) #返回非空白行和非测试行
    valid_df = get_df(valid_rel)
    #每个分组之后的每个文件的三维列表，文件名->行名->单词，还会返回每个文件到底是bug还是不是bug
    train_code3d, train_label = get_code3d_and_label(train_df, True)
    valid_code3d, valid_label = get_code3d_and_label(valid_df, True)
    #计算样本的权重 array([0.5498,5.5183])
    #它通过np.unique(train_label)获取训练数据集中的唯一类别 也就是class=（[true,flase]）
    #class 数据集的类别
    sample_weights = compute_class_weight(class_weight = 'balanced', classes = np.unique(train_label), y = train_label)
    #defect的权重是取大的，clean的权重是取小的
    weight_dict['defect'] = np.max(sample_weights)
    weight_dict['clean'] = np.min(sample_weights)
    
    w2v_dir = get_w2v_path() #word2vec_dir = '/hy-tmp/output/Word2Vec_model/' 
    #获取之前word2vec里面的文件
    word2vec_file_dir = os.path.join(w2v_dir,dataset_name+'-'+str(embed_dim)+'dim.bin')
    #加载预先训练好的模型
    word2vec = Word2Vec.load(word2vec_file_dir)
    print('load Word2Vec for',dataset_name,'finished')
    #为深度学习模型获取用于词嵌入的权重 做准备
    #返回的是一个tensor矩阵，形状是(词汇量大小, 词嵌入维度)。
    #传入的形参当中word2vec是一个词向量矩阵
    word2vec_weights = get_w2v_weight_for_deep_learning_models(word2vec, embed_dim)

    vocab_size = len(word2vec.wv.key_to_index)  + 1 # for unknown tokens
    #输出一个和train_code3d一样形状的三维列表
    x_train_vec = get_x_vec(train_code3d, word2vec)
    x_valid_vec = get_x_vec(valid_code3d, word2vec)
    #计算了 x_train_vec 中每个句子的长度，并将结果存储在一个列表中
    #max([len(sent) for sent in x_train_vec]): 这部分代码找到了列表中句子长度的最大值
    #最大值和LOC比较之后取 最小值
    #sent对应的是句子，也就是某一个文件中的每一行
    max_sent_len = min(max([len(sent) for sent in (x_train_vec)]), max_train_LOC)
    
    #train_label是一个列表，列表的每个元素对应的是activemq数据集中每个文件是不是true还是flase
    train_dl = get_dataloader(x_train_vec,train_label,batch_size,max_sent_len)

    valid_dl = get_dataloader(x_valid_vec, valid_label,batch_size,max_sent_len)

    model = HierarchicalAttentionNetwork(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        word_gru_hidden_dim=word_gru_hidden_dim,
        sent_gru_hidden_dim=sent_gru_hidden_dim,
        word_gru_num_layers=word_gru_num_layers,
        sent_gru_num_layers=sent_gru_num_layers,
        word_att_dim=word_att_dim,
        sent_att_dim=sent_att_dim,
        use_layer_norm=use_layer_norm,
        dropout=dropout)
    #将模型移动到GPU上，并且允许词嵌入层的权重在训练过程中进行更新。
    model = model.cuda()
    model.sent_attention.word_attention.freeze_embeddings(False)
    #optim.Adam是PyTorch中Adam优化器的类。它接收一个参数params，用于指定需要优化的模型参数
    #filter(lambda p: p.requires_grad, model.parameters())是一个过滤器，用于获取所有需要进行梯度更新的模型参数。
    #model.parameters()返回模型中所有的参数对象
    #p.requires_grad判断参数是否需要梯度更新，通过filter函数筛选出需要更新的参数
    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    #这是一个损失函数的对象
    criterion = nn.BCELoss() #二进制交叉熵损失函数，属于二分类问题，所以使用这样的损失函数
    #这段代码使用os.listdir()函数列出指定目录（actual_save_model_dir）中的文件，并将文件名存储在checkpoint_files列表中。
    checkpoint_files = os.listdir(actual_save_model_dir)
    #.ipynb_checkpoints是由Jupyter Notebook自动生成的文件夹。
    #当你在Jupyter Notebook中保存一个笔记本文件（.ipynb）时，Jupyter会在同一目录下创建一个名为.ipynb_checkpoints的文件夹。
    # 该文件夹用于存储笔记本的检查点信息，以便在需要时进行恢复或回滚。
    #检查点是在笔记本的编辑过程中自动创建的保存点。
    #它记录了笔记本在某个时间点的状态，包括所有单元格的代码、输出、变量等。
    #这样，如果你在编辑过程中出现错误或意外情况导致笔记本无法正常工作，你可以通过恢复到之前的一个检查点来回滚笔记本的状态，
    #避免丢失重要的工作进展。
    if '.ipynb_checkpoints' in checkpoint_files:
        checkpoint_files.remove('.ipynb_checkpoints')
    #训练完的数据之后的模型存放位置的文件个数多少
    total_checkpoints = len(checkpoint_files)

    # no model is trained 
    if total_checkpoints == 0:
        model.sent_attention.word_attention.init_embeddings(word2vec_weights)
        current_checkpoint_num = 1

        train_loss_all_epochs = []
        val_loss_all_epochs = []
    
    else:
        #re.findall('\d+', s)使用正则表达式提取s中的数字部分，并返回一个包含所有匹配项的列表。
        #[0]表示获取列表中的第一个元素，即提取到的数字部分
        checkpoint_nums = [int(re.findall('\d+',s)[0]) for s in checkpoint_files]
        current_checkpoint_num = max(checkpoint_nums)
        #torch.load()函数用于从磁盘中加载保存的模型或张量。它会读取指定文件的内容，
        #并返回一个包含模型状态字典（model state dictionary）的对象
        checkpoint = torch.load(actual_save_model_dir+'checkpoint_'+str(current_checkpoint_num)+'epochs.pth')
        #通过load_state_dict()方法
        #将模型的状态字典（model_state_dict）和优化器的状态字典（optimizer_state_dict）分别加载到模型和优化器中，
        #以恢复之前保存的训练状态
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        loss_df = pd.read_csv(loss_dir+dataset_name+'-loss_record.csv')
        train_loss_all_epochs = list(loss_df['train_loss'])
        val_loss_all_epochs = list(loss_df['valid_loss'])

        current_checkpoint_num = current_checkpoint_num+1 # go to next epoch
        print('continue training model from epoch',current_checkpoint_num)

    for epoch in tqdm(range(current_checkpoint_num,num_epochs+1)):
        train_losses = []
        val_losses = []

        model.train()

        for inputs, labels in train_dl:
            #inputs 三维张量   labels 一维张量
            inputs_cuda, labels_cuda = inputs.cuda(), labels.cuda()
            output, _, __, ___ = model(inputs_cuda)
            #注意看桌面上的及截图
            weight_tensor = get_loss_weight(labels)

            criterion.weight = weight_tensor

            loss = criterion(output, labels_cuda.reshape(batch_size,1))

            train_losses.append(loss.item())
            
            #用于清空GPU上的缓存空间。
            torch.cuda.empty_cache()
            
            #向后传播计算随机梯度的下降
            #在调用loss.backward()后，PyTorch会自动计算损失函数关于模型参数的梯度，
            #并将梯度存储在对应的参数的grad属性中。
            loss.backward()
            #梯度裁剪的函数，用于控制梯度爆炸或者梯度消失
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            #更新模型参数
            optimizer.step()

            torch.cuda.empty_cache()
        #取损失函数的平均值
        train_loss_all_epochs.append(np.mean(train_losses))

        with torch.no_grad():
            
            criterion.weight = None
            model.eval()
            
            for inputs, labels in valid_dl:

                inputs, labels = inputs.cuda(), labels.cuda()
                output, _, __, ___ = model(inputs)
            
                val_loss = criterion(output, labels.reshape(batch_size,1))

                val_losses.append(val_loss.item())

            val_loss_all_epochs.append(np.mean(val_losses))

        if epoch % save_every_epochs == 0:
            print(dataset_name,'- at epoch:',str(epoch))

            if exp_name == '':
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                            }, 
                            actual_save_model_dir+'checkpoint_'+str(epoch)+'epochs.pth')
            else:
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                            }, 
                            actual_save_model_dir+'checkpoint_'+exp_name+'_'+str(epoch)+'epochs.pth')

        loss_df = pd.DataFrame()
        loss_df['epoch'] = np.arange(1,len(train_loss_all_epochs)+1)
        loss_df['train_loss'] = train_loss_all_epochs
        loss_df['valid_loss'] = val_loss_all_epochs
        
        loss_df.to_csv(loss_dir+dataset_name+'-loss_record.csv',index=False)

dataset_name = args.dataset
train_model(dataset_name)