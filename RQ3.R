library(tidyverse)
library(gridExtra)
library(lattice)
library(ModelMetrics)
library(caret)
library(reshape2)
library(car)
library(carData)
library(pROC)
library(effsize)
library(ScottKnottESD)
library(dplyr)
library(tibble)

###### 将抽样提取到的代码行的所在文件全部放在一个目录下面 #####
###### 用undertsand的软件进行函数的圈复杂度等等统计 ######

restore_sample_file = function(df, model_name){
  
  df = select(df, test, filename, src)
  df = distinct(df)
  df$restorename = paste0(df$test,'*',df$filename)
  
  filename_and_test_list = df$restorename
  
  for(file_str in filename_and_test_list){

    file_filename = str_split_fixed(file_str, '\\*', 2)[,2]
    source.code = subset(df, select = c('filename','src'), subset = (filename == file_filename))
    cell_data = source.code[1,2]

    filename_change = gsub("/", "-", file_filename)
    
    # 分理处文件名对应的test名(考虑到不同的test里面存在相同的filename)
    file_testname = str_split_fixed(file_str, '\\*', 2)[,1]
    
    file_path = paste0('D:/Gitee-code/how-far-we-go-github项目提交 - 副本/随机抽样检查/',model_name, '随机抽样的文件/', file_testname, '-', filename_change)

    if (!file.exists(file_path)) {
      file.create(file_path)
    }
    writeLines(cell_data, file_path)
  }
  
}

############## 为的是利用GLANCE里面进行的tokens的统计数量 #####################
CEandNFCdir = "D:/Gitee-code/CLBI/result/Glance_MD_full_threshold/line_result/test/"

all_CEandNF_files = list.files(CEandNFCdir)

df_CEandNF_all <- NULL

for(f in all_CEandNF_files)
{
  df <- read.csv(paste0(CEandNFCdir, f))
  df$test = str_split_fixed(f, "-result", 2)[,1]
  df_CEandNF_all  <- rbind(df_CEandNF_all, df)
}

df_CEandNF_all = select(df_CEandNF_all, "predicted_buggy_lines", "predicted_buggy_line_numbers","test",'numbertokens')
names(df_CEandNF_all) = c("filename", "line.number","test","numbers.tokens")
df_CEandNF_all$filename = str_split_fixed(df_CEandNF_all$filename, ":", 2)[,1]


################ DeepLineDP 输出结果存放路径  ################################
prediction_dir = 'D:/Gitee-code/Boosting deep line-level defect prediction with syntactic features/all_models_result/within-release/'

all_files = list.files(prediction_dir)

df_all <- NULL

for(f in all_files)
{
  df <- read.csv(paste0(prediction_dir, f))
  df_all <- rbind(df_all, df)
}

############### 文件级数据集的存储位置 ##################
file_datasets_dir = 'D:/Gitee-code/CLBI/Dataset/File-level/'
all_file_level_files = list.files(file_datasets_dir)
df_dataset_files = NULL


for(f in all_file_level_files){
  df =  read.csv(paste0(file_datasets_dir, f))
  df$test = str_split_fixed(f, '_ground-truth-files_dataset', 2)[,1]
  df_dataset_files = rbind(df_dataset_files, df)
}

names(df_dataset_files) = c('filename', 'bug', 'src', 'test')



all_eval_releases = c('activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0', 
                      'camel-2.10.0', 'camel-2.11.0' , 
                      'derby-10.5.1.1' , 'groovy-1_6_BETA_2' , 'hbase-0.95.2', 
                      'hive-0.12.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1',  
                      'lucene-3.0.0', 'lucene-3.1', 'wicket-1.5.3')

line.ground.truth = select(df_all,  project, train, test, filename, file.level.ground.truth, prediction.prob, line.number, line.level.ground.truth, is.comment.line)
line.ground.truth = filter(line.ground.truth, file.level.ground.truth == "True" & prediction.prob >= 0.5 &  is.comment.line== "False")
line.ground.truth = distinct(line.ground.truth)

############################################################ DeepLineDP ######################################################
######## deeplinedp的总TP数目 ############
tp_deeplinedp_total = filter(line.ground.truth, line.level.ground.truth == 'True')

########## 按照14个版本的比例进行随机抽样 ############
sample_number_release = tp_deeplinedp_total %>% group_by(test, filename) %>% distinct(line.number)
sample_number_release = sample_number_release %>% group_by(test) %>% summarise(tp.numbers = n())
write.csv(sample_number_release, file= 'D:/Gitee-code/how-far-we-go-github项目提交 - 副本/DeepLineDP_TP总数(更新，筛选注释行).csv')

set.seed(123)

tp_deeplinedp_df = NULL

########## 由sample_number 确定每一个release上面的随机抽样的样本数量 ########
for(rel in all_eval_releases){
  temp = filter(tp_deeplinedp_total, test == rel);
  temp_sample = temp %>% sample_n(size = case_when(
    rel== "activemq-5.2.0" ~ 5,
    rel== "activemq-5.3.0" ~ 16,
    rel== "activemq-5.8.0" ~ 28,
    rel== "camel-2.10.0" ~ 14,
    rel== "camel-2.11.0" ~ 8,
    rel== "derby-10.5.1.1" ~ 30,
    rel== "groovy-1_6_BETA_2" ~ 8,
    rel== "hbase-0.95.2" ~ 173,
    rel== "hive-0.12.0" ~ 60,
    rel== "jruby-1.5.0" ~ 3,
    rel== "jruby-1.7.0.preview1" ~ 9,
    rel== "lucene-3.0.0" ~ 10,
    rel== "lucene-3.1" ~ 3,
    rel== "wicket-1.5.3" ~ 4,
    TRUE ~ 0 

  ))
  tp_deeplinedp_df = rbind(tp_deeplinedp_df, temp_sample)
}

######### 获取每个文件名对应的文件源代码 ##########
dp_result = merge(tp_deeplinedp_df, df_dataset_files, by = c('test', 'filename')) %>% select('test', 'filename', 'line.number', 'src')

# 获取每一行代码行的numbers of tokens 
dp_result = merge(dp_result, df_CEandNF_all, by = c('test','filename','line.number'), all.x = TRUE) %>%  select('test', 'filename','src','line.number','numbers.tokens')

### 将随机抽样的文件统一放在同一个文件夹下面
restore_sample_file(dp_result, "DeepLineDP")

# 获取每一个line.number在源文件中指定的代码行
dp_result = dp_result %>% mutate(line_content = mapply(function(src_line, line_num) readLines(textConnection(src_line))[line_num], src, line.number) 
                      ,total_lines = mapply(function(src_line) length(readLines(textConnection(src_line))), src)
                      ,line_Percent = line.number / total_lines)  %>%     
                      select('test', 'filename', 'line.number', 'line_content', 'numbers.tokens', 'line_Percent')
                       
write.csv(dp_result,file ='D:/Gitee-code/how-far-we-go-github项目提交 - 副本/随机抽样检查/DeepLineDP(update-lien-precent).csv')


############################################################ GLANCE ######################################################

glance.ea.result.dir = 'D:/Gitee-code/Boosting deep line-level defect prediction with syntactic features/all_models_result/BASE-Glance-EA/line_result/test/'

glance.md.result.dir = 'D:/Gitee-code/Boosting deep line-level defect prediction with syntactic features/all_models_result/BASE-Glance-MD/line_result/test/'

glance.lr.result.dir = 'D:/Gitee-code/Boosting deep line-level defect prediction with syntactic features/all_models_result/BASE-Glance-LR/line_result/test/'


all_EA_files = list.files(glance.ea.result.dir)
all_MD_files = list.files(glance.md.result.dir)
all_LR_files = list.files(glance.lr.result.dir)


df_ea_all <- NULL
df_md_all <- NULL
df_lr_all <- NULL

############################################################ GLANCE-EA ######################################################
### 获取EA ######
for(f in all_EA_files)
{
  df <- read.csv(paste0(glance.ea.result.dir, f))
  df$test = str_split_fixed(f, "-result", 2)[,1]
  df_ea_all  <- rbind(df_ea_all, df)
}

df_ea_all = select(df_ea_all, "predicted_buggy_lines","predicted_buggy_line_numbers", "test")
names(df_ea_all) = c("filename","line.number", "test")
df_ea_all$filename = str_split_fixed(df_ea_all$filename, ":", 2)[,1]
df_ea_all = distinct(df_ea_all)

temp_ea = merge(df_ea_all, line.ground.truth, by = c('filename', 'test', 'line.number'))
########### GLANCE-EA的总TP数目  ##################
tp_ea_total = filter(temp_ea, line.level.ground.truth == 'True')

########## 按照14个版本的比例进行随机抽样 ############
sample_number_release = tp_ea_total %>% group_by(test, filename) %>% distinct(line.number)
sample_number_release = sample_number_release %>% group_by(test) %>% summarise(tp.numbers = n())
write.csv(sample_number_release, file= 'D:/Gitee-code/how-far-we-go-github项目提交 - 副本/GLANCE-EA_TP总数.csv')


set.seed(123)

####### 每个release抽取的随机抽样样本都是由sample_number_release中可见 #########

tp_ea_df = NULL

for(rel in all_eval_releases){
  temp = filter(tp_ea_total, test == rel);
  temp_sample = temp %>% sample_n(size = case_when(
    rel== "activemq-5.2.0" ~ 5,
    rel== "activemq-5.3.0" ~ 14,
    rel== "activemq-5.8.0" ~ 30,
    rel== "camel-2.10.0" ~ 17,
    rel== "camel-2.11.0" ~ 8,
    rel== "derby-10.5.1.1" ~ 17,
    rel== "groovy-1_6_BETA_2" ~ 4,
    rel== "hbase-0.95.2" ~ 213,
    rel== "hive-0.12.0" ~ 39,
    rel== "jruby-1.5.0" ~ 2,
    rel== "jruby-1.7.0.preview1" ~ 4,
    rel== "lucene-3.0.0" ~ 6,
    rel== "lucene-3.1" ~ 3,
    rel== "wicket-1.5.3" ~ 3,
    TRUE ~ 0 

  ))
  tp_ea_df = rbind(tp_ea_df, temp_sample)
}


######### 获取每个文件名对应的文件源代码 ##########
ea_result = merge(tp_ea_df, df_dataset_files, by = c('test', 'filename')) %>% select('test', 'filename', 'line.number', 'src')

# 获取每一行代码行的numbers of tokens 
ea_result = merge(ea_result, df_CEandNF_all, by = c('test','filename','line.number'), all.x = TRUE) %>%  select('test', 'filename','src','line.number','numbers.tokens')

### 将随机抽样的文件统一放在同一个文件夹下面
restore_sample_file(ea_result, "GLANCE-EA")

# 获取每一个line.number在源文件中指定的代码行
ea_result = ea_result %>% mutate(line_content = mapply(function(src_line, line_num) readLines(textConnection(src_line))[line_num], src, line.number) 
                                 ,total_lines = mapply(function(src_line) length(readLines(textConnection(src_line))), src)
                                 ,line_Percent = line.number / total_lines)  %>%     
  select('test', 'filename', 'line.number', 'line_content', 'numbers.tokens', 'line_Percent')


write.csv(ea_result,file ='D:/Gitee-code/how-far-we-go-github项目提交 - 副本/随机抽样检查/GLANCE-EA(update_line_precent).csv')


############################################################ GLANCE-MD ######################################################

### 获取MD ######
for(f in all_MD_files)
{
  df <- read.csv(paste0(glance.md.result.dir, f))
  df$test = str_split_fixed(f, "-result", 2)[,1]
  df_md_all  <- rbind(df_md_all, df)
}

df_md_all = select(df_md_all, "predicted_buggy_lines","predicted_buggy_line_numbers","test")
names(df_md_all) = c("filename","line.number", "test")
df_md_all$filename = str_split_fixed(df_md_all$filename, ":", 2)[,1]
df_md_all = distinct(df_md_all)

temp_md = merge(df_md_all, line.ground.truth, by = c('filename', 'test', 'line.number'))
########### GLANCE-EA的总TP数目  ##################
tp_md_total = filter(temp_md, line.level.ground.truth == 'True')

########## 按照14个版本的比例进行随机抽样 ############
sample_number_release = tp_md_total %>% group_by(test, filename) %>% distinct(line.number)
sample_number_release = sample_number_release %>% group_by(test) %>% summarise(tp.numbers = n())
write.csv(sample_number_release, file= 'D:/Gitee-code/how-far-we-go-github项目提交 - 副本/GLANCE-MD_TP总数.csv')


set.seed(123)

####### 每个release抽取的随机抽样样本都是由sample_number_release中可见 #########
tp_md_df = NULL

for(rel in all_eval_releases){
  temp = filter(tp_md_total, test == rel);
  temp_sample = temp %>% sample_n(size = case_when(
    rel== "activemq-5.2.0" ~ 5,
    rel== "activemq-5.3.0" ~ 15,
    rel== "activemq-5.8.0" ~ 28,
    rel== "camel-2.10.0" ~ 15,
    rel== "camel-2.11.0" ~ 7,
    rel== "derby-10.5.1.1" ~ 27,
    rel== "groovy-1_6_BETA_2" ~ 8,
    rel== "hbase-0.95.2" ~ 179,
    rel== "hive-0.12.0" ~ 57,
    rel== "jruby-1.5.0" ~ 3,
    rel== "jruby-1.7.0.preview1" ~ 9,
    rel== "lucene-3.0.0" ~ 10,
    rel== "lucene-3.1" ~ 3,
    rel== "wicket-1.5.3" ~ 4,
    TRUE ~ 0 

  ))
  tp_md_df = rbind(tp_md_df, temp_sample)
}


######### 获取每个文件名对应的文件源代码 ##########
md_result = merge(tp_md_df, df_dataset_files, by = c('test', 'filename')) %>% select('test', 'filename', 'line.number', 'src')

### 将随机抽样的文件统一放在同一个文件夹下面
restore_sample_file(md_result, "GLANCE-MD")

# 获取每一行代码行的numbers of tokens 
md_result = merge(md_result, df_CEandNF_all, by = c('test','filename','line.number'), all.x = TRUE) %>%  select('test', 'filename','src','line.number','numbers.tokens')

# 获取每一个line.number在源文件中指定的代码行
md_result = md_result %>% mutate(line_content = mapply(function(src_line, line_num) readLines(textConnection(src_line))[line_num], src, line.number) 
                                 ,total_lines = mapply(function(src_line) length(readLines(textConnection(src_line))), src)
                                 ,line_Percent = line.number / total_lines)  %>%     
  select('test', 'filename', 'line.number', 'line_content', 'numbers.tokens', 'line_Percent')

write.csv(md_result,file ='D:/Gitee-code/how-far-we-go-github项目提交 - 副本/随机抽样检查/GLANCE-MD(update_line_precent).csv')

############################################################ GLANCE-LR ######################################################

### 获取LR ######
for(f in all_LR_files)
{
  df <- read.csv(paste0(glance.lr.result.dir, f))
  df$test = str_split_fixed(f, "-result", 2)[,1]
  df_lr_all  <- rbind(df_lr_all, df)
}

df_lr_all = select(df_lr_all, "predicted_buggy_lines","predicted_buggy_line_numbers","test")
names(df_lr_all) = c("filename","line.number", "test")
df_lr_all$filename = str_split_fixed(df_lr_all$filename, ":", 2)[,1]
df_lr_all = distinct(df_lr_all)


temp_lr = merge(df_lr_all, line.ground.truth, by = c('filename', 'test', 'line.number'))
########### GLANCE-LR的总TP数目  ##################
tp_lr_total = filter(temp_lr, line.level.ground.truth == 'True')

########## 按照14个版本的比例进行随机抽样 ############
sample_number_release = tp_lr_total %>% group_by(test, filename) %>% distinct(line.number)
sample_number_release = sample_number_release %>% group_by(test) %>% summarise(tp.numbers = n())
write.csv(sample_number_release, file= 'D:/Gitee-code/how-far-we-go-github项目提交 - 副本/GLANCE-LR_TP总数.csv')

set.seed(123)

####### 每个release抽取的随机抽样样本都是由sample_number_release中可见 #########
tp_lr_df = NULL

for(rel in all_eval_releases){
  temp = filter(tp_lr_total, test == rel);
  temp_sample = temp %>% sample_n(size = case_when(
    rel== "activemq-5.2.0" ~ 5,
    rel== "activemq-5.3.0" ~ 8,
    rel== "activemq-5.8.0" ~ 21,
    rel== "camel-2.10.0" ~ 11,
    rel== "camel-2.11.0" ~ 2,
    rel== "derby-10.5.1.1" ~ 32,
    rel== "groovy-1_6_BETA_2" ~ 5,
    rel== "hbase-0.95.2" ~ 239,
    rel== "hive-0.12.0" ~ 30,
    rel== "jruby-1.5.0" ~ 2,
    rel== "jruby-1.7.0.preview1" ~ 1,
    rel== "lucene-3.0.0" ~ 6,
    rel== "lucene-3.1" ~ 3,
    rel== "wicket-1.5.3" ~ 1,
    TRUE ~ 0 
  ))
  tp_lr_df = rbind(tp_lr_df, temp_sample)
}


######### 获取每个文件名对应的文件源代码 ##########
lr_result = merge(tp_lr_df, df_dataset_files, by = c('test', 'filename')) %>% select('test', 'filename', 'line.number', 'src')

# 获取每一行代码行的numbers of tokens 
lr_result = merge(lr_result, df_CEandNF_all, by = c('test','filename','line.number'), all.x = TRUE) %>%  select('test', 'filename','src','line.number','numbers.tokens')

### 将随机抽样的文件统一放在同一个文件夹下面
restore_sample_file(lr_result, "GLANCE-LR")

# 获取每一个line.number在源文件中指定的代码行
lr_result = lr_result%>% mutate(line_content = mapply(function(src_line, line_num) readLines(textConnection(src_line))[line_num], src, line.number) 
                                ,total_lines = mapply(function(src_line) length(readLines(textConnection(src_line))), src)
                                ,line_Percent = line.number / total_lines)  %>%     
  select('test', 'filename', 'line.number', 'line_content', 'numbers.tokens', 'line_Percent')


write.csv(lr_result,file ='D:/Gitee-code/how-far-we-go-github项目提交 - 副本/随机抽样检查/GLANCE-LR(update_line_precent).csv')


