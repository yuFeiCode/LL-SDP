#2023-10-14修订

library(tidyverse)
library(gridExtra)

library(ModelMetrics)

library(caret)

library(reshape2)
library(pROC)

library(effsize)
library(ScottKnottESD)

library(dplyr)
library(tibble)

save.fig.dir = 'E:/mypapers/yufei/defect prediction/(2023-09-03)lineDP how far/result/figure/'

dir.create(file.path(save.fig.dir), showWarnings = FALSE)

preprocess <- function(x, reverse){
  colnames(x) <- c("variable","value")
  tmp <- do.call(cbind, split(x, x$variable))
  tmp <- tmp[, grep("value", names(tmp))]
  names(tmp) <- gsub(".value", "", names(tmp))
  df <- tmp
  ranking <- NULL
  
  if(reverse == TRUE)
  { 
    ranking <- (max(sk_esd(df)$group)-sk_esd(df)$group) +1 
  }
  else
  { 
    ranking <- sk_esd(df)$group 
  }
  
  x$rank <- paste("Rank",ranking[as.character(gsub("-", ".", x$variable))])
  return(x)
}

get.top.k.tokens = function(df, k)
{
  top.k <- df %>% filter( is.comment.line=="False"  & file.level.ground.truth=="True" & prediction.label=="True" ) %>%
    group_by(test, filename) %>% top_n(k, token.attention.score) %>% select("project","train","test","filename","token") %>% distinct()
  
  top.k$flag = 'topk'

  return(top.k)
}


prediction_dir = 'E:/mypapers/yufei/defect prediction/(2023-09-03)lineDP how far/result/DeepLineDP实验输出结果/within-release/'


all_files = list.files(prediction_dir)

df_all <- NULL

for(f in all_files)
{
  df <- read.csv(paste0(prediction_dir, f))
  df_all <- rbind(df_all, df)
}

# ---------------- Code for RQ1 ----------------------- #

get.file.level.metrics = function(df.file, method.name)
{
   all.gt = as.factor(df.file$oracle)
   all.prob = df.file$predicted_score
   all.pred = as.factor(df.file$predicted_label)

   if (method.name == "DeepLineDP")
   {
      all.gt = as.factor(df.file$file.level.ground.truth)
      all.prob = df.file$prediction.prob
      all.pred = as.factor(df.file$prediction.label)
    }
  
   confusion.mat = confusionMatrix(all.pred, reference = all.gt)
  
   bal.acc = confusion.mat$byClass["Balanced Accuracy"]
   AUC = pROC::auc(all.gt, all.prob)
  
   levels(all.pred)[levels(all.pred)=="False"] = 0
   levels(all.pred)[levels(all.pred)=="True"] = 1
   levels(all.gt)[levels(all.gt)=="False"] = 0
   levels(all.gt)[levels(all.gt)=="True"] = 1
  
   all.gt = as.numeric_version(all.gt)
   all.gt = as.numeric(all.gt)
  
   all.pred = as.numeric_version(all.pred)
   all.pred = as.numeric(all.pred)
  
   MCC = mcc(all.gt, all.pred, cutoff = 0.5) 
  
   if(is.nan(MCC))
   {
     MCC = 0
   }
  
   eval.result = c(AUC, MCC, bal.acc)
  
  return(eval.result)
}

get.file.level.eval.result = function(prediction.dir, method.name)
{
  all_files = list.files(prediction.dir)

  all.auc = c()
  all.mcc = c()
  all.bal.acc = c()
  all.test.rels = c()

  for(f in all_files) # for looping through files
  {
     df = read.csv(paste0(prediction.dir, f))

     if (method.name == "DeepLineDP")
     {
       df = as_tibble(df)
       df = select(df, c(filename, file.level.ground.truth, prediction.prob, prediction.label))    
       df = distinct(df)
     }else{
       df = as_tibble(df)
       df = select(df, c(filename, oracle, predicted_score, predicted_label))      
       df = distinct(df)
    }

    file.level.result = get.file.level.metrics(df, method.name)

    AUC = file.level.result[1]
    MCC = file.level.result[2]
    bal.acc = file.level.result[3]

    all.auc = append(all.auc,AUC)
    all.mcc = append(all.mcc,MCC)
    all.bal.acc = append(all.bal.acc,bal.acc)
    all.test.rels = append(all.test.rels,f)
  }
  
  result.df = data.frame(all.auc,all.mcc,all.bal.acc)
  
  all.test.rels = str_replace(all.test.rels, ".csv", "")
  
  result.df$release = all.test.rels
  result.df$technique = method.name
  
  return(result.df)
}
 

deeplinedp.prediction.dir = "E:/mypapers/yufei/defect prediction/(2023-09-03)lineDP how far/result/DeepLineDP实验输出结果/within-release/" 
deepline.dp.result = get.file.level.eval.result(deeplinedp.prediction.dir, "DeepLineDP")

GLANCE.EA.prediction.dir = "E:/mypapers/yufei/defect prediction/(2023-09-03)lineDP how far/result/(2023-10-13)glance/line-threshold=1__BASE-Glance-ALL(2023-10-13)/BASE-Glance-EA/file_result/test/"
GLANCE.ea.result = get.file.level.eval.result(GLANCE.EA.prediction.dir, "GLANCE-EA")

GLANCE.MD.prediction.dir = "E:/mypapers/yufei/defect prediction/(2023-09-03)lineDP how far/result/(2023-10-13)glance/line-threshold=1__BASE-Glance-ALL(2023-10-13)/BASE-Glance-MD/file_result/test/"
GLANCE.md.result = get.file.level.eval.result(GLANCE.MD.prediction.dir, "GLANCE-MD")

GLANCE.LR.prediction.dir = 'E:/mypapers/yufei/defect prediction/(2023-09-03)lineDP how far/result/(2023-10-13)glance/line-threshold=1__BASE-Glance-ALL(2023-10-13)/BASE-Glance-LR/file_result/test/'
GLANCE.lr.result = get.file.level.eval.result(GLANCE.LR.prediction.dir, "GLANCE-LR")

all.result = rbind(GLANCE.lr.result, GLANCE.ea.result, GLANCE.md.result, deepline.dp.result)

names(all.result) = c("AUC","MCC","Balance.Accuracy","Release", "Technique")

auc.result = select(all.result, c("Technique","AUC"))
auc.result = preprocess(auc.result,FALSE)
auc.result[auc.result$variable=="GLANCE-LR", "variable"] = "GLANCE-LR"

mcc.result = select(all.result, c("Technique","MCC"))
mcc.result = preprocess(mcc.result,FALSE)
mcc.result[mcc.result$variable=="GLANCELR", "variable"] = "GLANCELR"

bal.acc.result = select(all.result, c("Technique","Balance.Accuracy"))
bal.acc.result = preprocess(bal.acc.result,FALSE)
bal.acc.result[bal.acc.result$variable=="GLANCE-LR", "variable"] = "GLANCE-LR"

ggplot(auc.result, aes(x=reorder(variable, -value, FUN=median), y=value)) + geom_boxplot() + facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + ylab("AUC") + xlab("")+ theme(axis.text.x=element_text(angle = -45, hjust = 0))
ggsave(paste0(save.fig.dir,"file-AUC.pdf"),width=5,height=2.5)

ggplot(bal.acc.result, aes(x=reorder(variable, value, FUN=median), y=value)) + geom_boxplot() + facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + ylab("Balance Accuracy") + xlab("")+ theme(axis.text.x=element_text(angle = -45, hjust = 0))
ggsave(paste0(save.fig.dir,"file-Balance_Accuracy.pdf"),width=5,height=2.5)

ggplot(mcc.result, aes(x=reorder(variable, value, FUN=median), y=value)) + geom_boxplot()  + facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + ylab("MCC") + xlab("")+ theme(axis.text.x=element_text(angle = -45, hjust = 0))
ggsave(paste0(save.fig.dir, "file-MCC.pdf"),width=5,height=2.5)
 

# ---------------- Code for RQ2 -----------------------#
## 两阶段预测，第一阶段文件级别，第二阶段行级。在第一阶段，人工审查或者测试那些被预测为有缺陷的文件，知道了哪些文件真
## 的有缺陷；在第二阶段，人工审查上一段得到的真的有缺陷的文件，看哪些行有缺陷，是针对每个实际有缺陷的文件进行行级缺陷
## 预测，因此在每个实际有缺陷的文件上计算行级性能指标

## 在DeepLineDP论文中，对于行级的baseline models，只考虑一个阶段，
## 直接对test版本进行行级缺陷预测（不像GLANCE的做法，先对文件预测，再对预测为有缺陷的文件进行行级预测），
## 即对test版本中所有实际有缺陷的文件进行行级预测；在与DeepLineDP进行比较时，
## 只取那些被DeepLineDP预测为有缺陷且实际有缺陷的文件进行行级预测比较
## 因此，DeepLineDP和baseline models是配对的实验结果

## 在RQ2中，我们采用DeepLineDP的评价方法来评价GLANCE：取阈值threshold=1，对预测为有缺陷的文件而言，所有的代码行都参与排序；针对那些被预测为有缺陷且实际有缺陷的文件，分析代码行级的排序性能指标IFA、recall@20%loc、effort@20%recall


## prepare data for baseline
line.ground.truth = select(df_all,  project, train, test, filename, file.level.ground.truth, prediction.prob, line.number, line.level.ground.truth, is.comment.line)
line.ground.truth = filter(line.ground.truth, file.level.ground.truth == "True" & prediction.prob >= 0.5 &  is.comment.line== "False")
line.ground.truth = distinct(line.ground.truth)

get.line.metrics.result = function(baseline.df, cur.df.file)
{
    #去掉glance输出的文件名后的":行号"
    baseline.df$filename = str_split_fixed(baseline.df$filename, ":", 2)[,1]
    baseline.df.with.ground.truth = merge(baseline.df, cur.df.file, by=c("filename", "line.number"))

    ## 同一文件内的行为一组，按line.score从大到小降序排列；每一组内独立编号order
    sorted = baseline.df.with.ground.truth %>% group_by(filename) %>% arrange(rank, .by_group = TRUE) %>% mutate(order = row_number())%>% mutate(totalSLOC = n())

  
    ## IFA:  每个文件一个IFA，在每个文件的行组中取order最低的行号，代表检查到第一个有缺陷的行时需要检查多少行
    IFA = sorted %>% filter(line.level.ground.truth == "True") %>% group_by(filename)  %>% top_n(1, -order)
    ## added 2023-09-16 确保按文件名排序
    IFA = IFA%>% arrange(filename)

    ## 注意要减1，第一个有缺陷语句前面的clean语句行数
    ifa.list = IFA$order - 1
  
    ##统计每个有缺陷的文件中包含多少个有缺陷的代码行
    total_true = sorted %>%  group_by(filename) %>% summarize(total_true = sum(line.level.ground.truth == "True"))

    ##glance的预测结果中包含file.level.ground.truth == "FALSE"的文件，
    ##为使用DeepLineDP使用的性能指标，这些文件需要排除掉
    total_true = total_true %>% filter(total_true > 0)

    ## added 2023-09-16 确保按文件名排序
    total_true = total_true%>% arrange(filename)
  
    #Recall20%LOC: 按行line.score从大到小排好序后，计算前20%LOC对应的行级缺陷召回率
    recall20LOC = sorted %>% group_by(filename) %>% mutate(effort = round(order/n(),digits = 2 )) %>% filter(effort <= 0.2) %>%
    summarize(correct_pred = sum(line.level.ground.truth == "True")) %>%
    merge(total_true) %>% mutate(recall20LOC = correct_pred/total_true)

    ## added 2023-09-16 确保按文件名排序
    recall20LOC = recall20LOC %>% arrange(filename)
  
    recall.list = recall20LOC$recall20LOC

    #Effort20%Recall：按行line.score从大到小排好序后，计算达到20%行级缺陷召回率时所需要审查的代码行占比，即effort
    effort20Recall = sorted %>% merge(total_true) %>% group_by(filename) %>% mutate(cummulative_correct_pred = cumsum(line.level.ground.truth == "True"), recall = round(cumsum(line.level.ground.truth == "True")/total_true, digits = 2)) %>%
    summarise(effort20Recall = sum(recall <= 0.2)/n())

    ## added 2023-09-16 确保按文件名排序
    effort20Recall = effort20Recall %>% arrange(filename)
  
    effort.list = effort20Recall$effort20Recall

    ## 按文件名排序后，才能保证每行是相同文件的ifa、recall和effort
    result.df = data.frame(IFA$filename, ifa.list, recall.list, effort.list, IFA$totalSLOC)
  
    return(result.df)
}

all_eval_releases = c('activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0', 
                      'camel-2.10.0', 'camel-2.11.0' , 
                      'derby-10.5.1.1' , 'groovy-1_6_BETA_2' , 'hbase-0.95.2', 
                      'hive-0.12.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1',  
                      'lucene-3.0.0', 'lucene-3.1', 'wicket-1.5.3')

glance.ea.result.dir = 'E:/mypapers/yufei/defect prediction/(2023-09-03)lineDP how far/result/(2023-10-13)glance/threshold=1_GLANCE-ALL(2023-10-13)/BASE-Glance-EA/line_result/test/'

glance.md.result.dir = 'E:/mypapers/yufei/defect prediction/(2023-09-03)lineDP how far/result/(2023-10-13)glance/threshold=1_GLANCE-ALL(2023-10-13)/BASE-Glance-MD/line_result/test/'

glance.lr.result.dir = 'E:/mypapers/yufei/defect prediction/(2023-09-03)lineDP how far/result/(2023-10-13)glance/threshold=1_GLANCE-ALL(2023-10-13)/BASE-Glance-LR/line_result/test/'

glance.ea.result.df = NULL
glance.md.result.df = NULL
glance.lr.result.df = NULL 

## get result from baseline
for(rel in all_eval_releases)
{  

    glance.ea.result = read.csv(paste0(glance.ea.result.dir,rel,'-result.csv'))
    glance.ea.result = select(glance.ea.result, "predicted_buggy_lines", "predicted_buggy_line_numbers","rank")
    names(glance.ea.result) = c("filename", "line.number", "rank")

    glance.md.result = read.csv(paste0(glance.md.result.dir,rel,'-result.csv'))
    glance.md.result = select(glance.md.result, "predicted_buggy_lines", "predicted_buggy_line_numbers","rank")
    names(glance.md.result) = c("filename", "line.number", "rank")
 
    glance.lr.result = read.csv(paste0(glance.lr.result.dir,rel,'-result.csv'))
    glance.lr.result = select(glance.lr.result, "predicted_buggy_lines", "predicted_buggy_line_numbers","rank")
    names(glance.lr.result) = c("filename", "line.number", "rank")

    cur.df.file = filter(line.ground.truth, test==rel)
    cur.df.file = select(cur.df.file, filename, line.number, line.level.ground.truth)

     ##"%>% mutate(test=rel)" 确保记录下每个target project的名称
     glance.ea.eval.result = get.line.metrics.result(glance.ea.result, cur.df.file) %>% mutate(test=rel)
     glance.md.eval.result = get.line.metrics.result(glance.md.result, cur.df.file) %>% mutate(test=rel)
     glance.lr.eval.result = get.line.metrics.result(glance.lr.result, cur.df.file) %>% mutate(test=rel)

     glance.ea.result.df = rbind(glance.ea.result.df, glance.ea.eval.result)
     glance.md.result.df = rbind(glance.md.result.df, glance.md.eval.result)
     glance.lr.result.df = rbind(glance.lr.result.df, glance.lr.eval.result)

     print(paste0('finished ', rel))
}


#Force attention score of comment line is 0
df_all[df_all$is.comment.line == "True",]$token.attention.score = 0

tmp.top.k = get.top.k.tokens(df_all, 1500)

merged_df_all = merge(df_all, tmp.top.k, by=c('project', 'train', 'test', 'filename', 'token'), all.x = TRUE)

merged_df_all[is.na(merged_df_all$flag),]$token.attention.score = 0

## use top-k tokens 
sum_line_attn = merged_df_all %>% filter(file.level.ground.truth == "True" & prediction.label == "True" ) %>% group_by(test, filename,is.comment.line, file.level.ground.truth, prediction.label, line.number, line.level.ground.truth) %>%
  summarize(attention_score = sum(token.attention.score), num_tokens = n())

##非常重要：需要filter(is.comment.line== "False") ，原始论文中没有过滤，导致注释行参与排序和性能的计算，放大了recall，降低了effort
sorted = sum_line_attn %>% filter(is.comment.line== "False") %>% group_by(test, filename) %>%  arrange(-attention_score, .by_group=TRUE) %>% mutate(order = row_number())%>% mutate(totalSLOC = n())

## get result from DeepLineDP
# calculate IFA
IFA = sorted %>% filter(line.level.ground.truth == "True") %>% group_by(test, filename) %>% top_n(1, -order)
 
## added 2023-09-16 确保按文件名排序
IFA = IFA %>% arrange(test, filename)

total_true = sorted %>% group_by(test, filename) %>% summarize(total_true = sum(line.level.ground.truth == "True"))
total_true = total_true %>% filter(total_true > 0)

## added 2023-09-16 确保按文件名排序
total_true = total_true %>% arrange(test, filename)

# calculate Recall20%LOC
recall20LOC = sorted %>% group_by(test, filename) %>% mutate(effort = round(order/n(),digits = 2 )) %>% filter(effort <= 0.2) %>%
  summarize(correct_pred = sum(line.level.ground.truth == "True")) %>%
  merge(total_true) %>% mutate(recall20LOC = correct_pred/total_true)

## added 2023-09-16 确保按文件名排序
recall20LOC = recall20LOC %>% arrange(test, filename)

# calculate Effort20%Recall
effort20Recall = sorted %>% merge(total_true) %>% group_by(test, filename) %>% mutate(cummulative_correct_pred = cumsum(line.level.ground.truth == "True"), recall = round(cumsum(line.level.ground.truth == "True")/total_true, digits = 2)) %>%
  summarise(effort20Recall = sum(recall <= 0.2)/n())

## added 2023-09-16 确保按文件名排序
effort20Recall = effort20Recall %>% arrange(test, filename)

## prepare data for plotting
deeplinedp.ifa = IFA$order - 1
deeplinedp.recall = recall20LOC$recall20LOC
deeplinedp.effort = effort20Recall$effort20Recall

deepline.dp.line.result = data.frame(IFA$filename, deeplinedp.ifa, deeplinedp.recall, deeplinedp.effort, IFA$test, IFA$totalSLOC)

glance.lr.result.df = glance.lr.result.df %>%   summarize(IFA=median(ifa.list),recall=median(recall.list),effort=median(effort.list), .by=test)
glance.ea.result.df = glance.ea.result.df %>%   summarize(IFA=median(ifa.list),recall=median(recall.list),effort=median(effort.list), .by=test)
glance.md.result.df = glance.md.result.df %>%   summarize(IFA=median(ifa.list),recall=median(recall.list),effort=median(effort.list), .by=test)
deepline.dp.line.result = deepline.dp.line.result %>%   summarize(IFA=median(deeplinedp.ifa),recall=median(deeplinedp.recall),effort=median(deeplinedp.effort), .by=IFA.test)

names(glance.lr.result.df) = c("release", "IFA", "Recall20%LOC", "Effort@20%Recall")
names(glance.ea.result.df) = c("release", "IFA", "Recall20%LOC", "Effort@20%Recall")
names(glance.md.result.df)  = c("release", "IFA", "Recall20%LOC", "Effort@20%Recall")
names(deepline.dp.line.result) = c("release", "IFA", "Recall20%LOC", "Effort@20%Recall")


glance.lr.result.df$technique = 'GLANCE-LR'
glance.ea.result.df$technique = 'GLANCE-EA'
glance.md.result.df$technique = 'GLANCE-MD'
deepline.dp.line.result$technique = 'DeepLineDP'


all.line.result = rbind(glance.lr.result.df, glance.ea.result.df, glance.md.result.df, deepline.dp.line.result)

recall.result.df = select(all.line.result, c('technique', 'Recall20%LOC'))
ifa.result.df = select(all.line.result, c('technique', 'IFA'))
effort.result.df = select(all.line.result, c('technique', 'Effort@20%Recall'))

recall.result.df = preprocess(recall.result.df, FALSE)
ifa.result.df = preprocess(ifa.result.df, TRUE)
effort.result.df = preprocess(effort.result.df, TRUE)

ggplot(recall.result.df, aes(x=reorder(variable, -value, FUN=median), y=value)) + geom_boxplot() + facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + ylab("Recall@Top20%LOC") + xlab("") + theme(axis.text.x=element_text(angle = -45, hjust = 0))
ggsave(paste0(save.fig.dir,"file-Recall@Top20LOC.pdf"),width=5,height=2.5)

ggplot(effort.result.df, aes(x=reorder(variable, value, FUN=median), y=value)) + geom_boxplot() + facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + ylab("Effort@Top20%Recall") + xlab("")+ theme(axis.text.x=element_text(angle = -45, hjust = 0))
ggsave(paste0(save.fig.dir,"file-Effort@Top20Recall.pdf"),width=5,height=2.5)

ggplot(ifa.result.df, aes(x=reorder(variable, value, FUN=median), y=value)) + geom_boxplot()  + coord_cartesian(ylim=c(0,175)) + facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + ylab("IFA") + xlab("")+ theme(axis.text.x=element_text(angle = -45, hjust = 0))
ggsave(paste0(save.fig.dir, "file-IFA.pdf"),width=5,height=2.5)

 
