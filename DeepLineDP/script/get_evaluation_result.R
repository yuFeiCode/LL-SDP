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

save.fig.dir = 'D:/Gitee-code/DeepLineDP_2.0/output/figure/'

dir.create(file.path(save.fig.dir), showWarnings = FALSE)

preprocess <- function(x, reverse){ #auc.result = select(all.result, c("Technique","AUC"))
                                    #auc.result = preprocess(auc.result,FALSE)
                                    # Technique指的是model.name
  colnames(x) <- c("variable","value")
  tmp <- do.call(cbind, split(x, x$variable))
  tmp <- tmp[, grep("value", names(tmp))] #grep()用来匹配列名包含value的列，返回一个像向量，包含匹配到的列的下标
  names(tmp) <- gsub(".value", "", names(tmp))
  df <- tmp
  ranking <- NULL
  
  if(reverse == TRUE)
  { 
    ranking <- (max(sk_esd(df)$group)-sk_esd(df)$group) +1 
  }
  else
  { 
    #自定义的 sk_esd() 函数接受一个数据 x，
    #执行方差分析并应用 Scott-Knott 检验来进行异常值检测。
    #最终返回一个带有异常值组信息的结果对象
    ranking <- sk_esd(df)$group 
  }
  
  x$rank <- paste("Rank",ranking[as.character(x$variable)])
  return(x)
}

get.top.k.tokens = function(df, k)
{
  top.k <- df %>% filter( is.comment.line=="False"  & file.level.ground.truth=="True" & prediction.label=="True" ) %>%
    group_by(test, filename) %>% top_n(k, token.attention.score) %>% select("project","train","test","filename","token") %>% distinct()
  
  top.k$flag = 'topk'

  return(top.k)
}


prediction_dir1 = 'D:/Gitee-code/DeepLineDP_2.0/output/prediction/DeepLineDP/within-release/'

prediction_dir = 'D:/Gitee-code/DeepLineDP_2.0/output/prediction/GLANCE/GLANCE_LR/File_Level/'

all_files = list.files(prediction_dir1)

df_all <- NULL

for(f in all_files)
{
  df <- read.csv(paste0(prediction_dir1, f))
  df_all <- rbind(df_all, df)
}

# # ---------------- Code for RQ1 -----------------------#
# 
# #RQ1-1
# df.to.plot = df_all %>% filter(is.comment.line=="False" & file.level.ground.truth=="True" & prediction.label=="True") %>% group_by(test, filename,token) %>%
#   summarise(Range=max(token.attention.score)-min(token.attention.score), SD=sd(token.attention.score)) %>%
#   melt() 
# 
# df.to.plot %>% ggplot(aes(x=variable, y=value)) + geom_boxplot() + scale_y_continuous(breaks=0:4*0.25) + xlab("") + ylab("")
# 
# ggsave(paste0(save.fig.dir,"rq1-1.pdf"),width=2.5,height=2.5)
# 
# 
# #RQ1-2
# 
# df_all_copy = data.frame(df_all)
# 
# df_all_copy = filter(df_all_copy, is.comment.line=="False" & file.level.ground.truth=="True" & prediction.label=="True")
# 
# clean.lines.df = filter(df_all_copy, line.level.ground.truth=="False")
# buggy.line.df = filter(df_all_copy, line.level.ground.truth=="True")
# 
# clean.lines.token.score = clean.lines.df %>% group_by(test, filename, token) %>% summarise(score = min(token.attention.score))
# clean.lines.token.score$class = "Clean Lines"
# 
# buggy.lines.token.score = buggy.line.df %>% group_by(test, filename, token) %>% summarise(score = max(token.attention.score))
# buggy.lines.token.score$class = "Defective Lines"
# 
# all.lines.token.score = rbind(buggy.lines.token.score, clean.lines.token.score)
# all.lines.token.score$class = factor(all.lines.token.score$class, levels = c('Defective Lines', 'Clean Lines'))
# 
# all.lines.token.score %>% ggplot(aes(x=class, y=score)) + geom_boxplot() + xlab("")  + ylab("Riskiness Score") 
# ggsave(paste0(save.fig.dir,"rq1-2.pdf"),width=2.5,height=2.5)
# 
# res = cliff.delta(buggy.lines.token.score$score, clean.lines.token.score$score)


# # ---------------- Code for RQ2 -----------------------#
# 
# get.file.level.metrics = function(df.file, method.name)
# {
#   
# 
#    if (method.name == "DeepLineDP")
#    {
#       all.gt = as.factor(df.file$file.level.ground.truth)
#       all.prob = df.file$prediction.prob
#       all.pred = as.factor(df.file$prediction.label)
#      }else {
#         all.gt = as.factor(df.file$oracle) #将提取出来的列变成因子类型，备注：oracle == file.level.ground.truth
#         all.prob = df.file$predicted_score
#         all.pred = as.factor(df.file$predicted_label)
#      }
#   
#   confusion.mat = confusionMatrix(all.pred, reference = all.gt)
#   
#   bal.acc = confusion.mat$byClass["Balanced Accuracy"] #混淆矩阵提供了评估分类模型性能的各种指标，如准确度（accuracy）、召回率（recall）等。
#   AUC = pROC::auc(all.gt, all.prob)
#   
#   levels(all.pred)[levels(all.pred)=="False"] = 0
#   levels(all.pred)[levels(all.pred)=="True"] = 1
#   levels(all.gt)[levels(all.gt)=="False"] = 0
#   levels(all.gt)[levels(all.gt)=="True"] = 1
#   
#   all.gt = as.numeric_version(all.gt)
#   all.gt = as.numeric(all.gt)
#   
#   all.pred = as.numeric_version(all.pred)
#   all.pred = as.numeric(all.pred)
#   
#   MCC = mcc(all.gt, all.pred, cutoff = 0.5) 
#   
#   if(is.nan(MCC))
#   {
#     MCC = 0
#   }
#   
#   eval.result = c(AUC, MCC, bal.acc)
#   
#   return(eval.result)
# }
# 
# get.file.level.eval.result = function(prediction.dir, method.name)
# {
#   all_files = list.files(prediction.dir)
# 
#   all.auc = c()
#   all.mcc = c()
#   all.bal.acc = c()
#   all.test.rels = c() #存放已经测试过的文件
# 
#   for(f in all_files) # for looping through files
#   {
#     df = read.csv(paste0(prediction.dir, f))
# 
# 
# 
#      
#      if (method.name == "DeepLineDP")
#      {
#       df = as_tibble(df)
#       df = select(df, c(filename, file.level.ground.truth, prediction.prob, prediction.label))
#       
#       df = distinct(df)
#      }else{
#      df = as_tibble(df)
#      df = select(df, c(filename, oracle, predicted_score, predicted_label))
#       
#      df = distinct(df)
#     }
# 
#     
#     
#     file.level.result = get.file.level.metrics(df, method.name)
# 
#     AUC = file.level.result[1]
#     MCC = file.level.result[2]
#     bal.acc = file.level.result[3] #file.level.result中存放的是单个文件（actvemq5.0.0）得出的文件级分类指标(AUC,MCC,Blacce Auc)
# 
#     all.auc = append(all.auc,AUC)
#     all.mcc = append(all.mcc,MCC)
#     all.bal.acc = append(all.bal.acc,bal.acc)
#     all.test.rels = append(all.test.rels,f)
# 
#   }
#   
#   result.df = data.frame(all.auc,all.mcc,all.bal.acc)
# 
#   
#   all.test.rels = str_replace(all.test.rels, ".csv", "")
#   
#   result.df$release = all.test.rels # $ 可以创建一个新的列名为relese，且将其赋值为all.test.rels
#   result.df$technique = method.name
#   
#   return(result.df)
# }
#  
# 
# deeplinedp.prediction.dir = "D:/Gitee-code/DeepLineDP_2.0/output/prediction/DeepLineDP/within-release/" 
# deepline.dp.result = get.file.level.eval.result(deeplinedp.prediction.dir, "DeepLineDP")
# 
# GLANCE.EA.prediction.dir = "D:/Gitee-code/DeepLineDP_2.0/output/prediction/GLANCE/GLANCE_EA/File_Level/"
# GLANCE.ea.result = get.file.level.eval.result(GLANCE.EA.prediction.dir, "GLANCEEA")
# 
# GLANCE.MD.prediction.dir = "D:/Gitee-code/DeepLineDP_2.0/output/prediction/GLANCE/GLANCE_MD/File_Level/"
# GLANCE.md.result = get.file.level.eval.result(GLANCE.MD.prediction.dir, "GLANCEMD")
# 
# GLANCE.lr.result = get.file.level.eval.result(prediction_dir, "GLANCELR")
# 
# all.result = rbind(GLANCE.lr.result, GLANCE.ea.result, GLANCE.md.result, deepline.dp.result)
# 
# names(all.result) = c("AUC","MCC","Balance.Accuracy","Release", "Technique")
# #可以在这里面保存一下csv文件
# auc.result = select(all.result, c("Technique","AUC"))
# auc.result = preprocess(auc.result,FALSE)
# auc.result[auc.result$variable=="GLANCELR", "variable"] = "GLANCELR"#将满足条件 auc.result$variable=="GLANCELR" 的行的 variable 列的值修改为 "GLANCELR"。
# 
# mcc.result = select(all.result, c("Technique","MCC"))
# mcc.result = preprocess(mcc.result,FALSE)
# mcc.result[mcc.result$variable=="GLANCELR", "variable"] = "GLANCELR"
# 
# bal.acc.result = select(all.result, c("Technique","Balance.Accuracy"))
# bal.acc.result = preprocess(bal.acc.result,FALSE)
# bal.acc.result[bal.acc.result$variable=="GLANCELR", "variable"] = "GLANCELR"
# 
#  
# ggplot(auc.result, aes(x=reorder(variable, -value, FUN=median), y=value)) + geom_boxplot() + facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + ylab("AUC") + xlab("")
# ggsave(paste0(save.fig.dir,"file-AUC.pdf"),width=4,height=2.5)
# 
# ggplot(bal.acc.result, aes(x=reorder(variable, value, FUN=median), y=value)) + geom_boxplot() + facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + ylab("Balance Accuracy") + xlab("")
# ggsave(paste0(save.fig.dir,"file-Balance_Accuracy.pdf"),width=4,height=2.5)
# 
# ggplot(mcc.result, aes(x=reorder(variable, value, FUN=median), y=value)) + geom_boxplot()  + facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + ylab("MCC") + xlab("")
# ggsave(paste0(save.fig.dir, "file-MCC.pdf"),width=4,height=2.5)
 

 # ---------------- Code for RQ3 -----------------------#
 ## 两阶段预测，第一阶段文件级别，第二阶段行级。在第一阶段，人工审查或者测试那些被预测为有缺陷的文件，知道了哪些文件真
 ## 的有缺陷；在第二阶段，人工审查上一段得到的真的有缺陷的文件，看哪些行有缺陷，是针对每个实际有缺陷的文件进行行级缺陷
 ## 预测，因此在每个实际有缺陷的文件上计算行级性能指标
 
 ## 在DeepLineDP论文中，对于行级的baseline models，只考虑一个阶段，
 ## 直接对test版本进行行级缺陷预测（不像GLANCE的做法，先对文件预测，再对预测为有缺陷的文件进行行级预测），
 ## 即对test版本中所有实际有缺陷的文件进行行级预测；在与DeepLineDP进行比较时，
 ## 只取那些被DeepLineDP预测为有缺陷且实际有缺陷的文件进行行级预测比较
 ## 因此，DeepLineDP和baseline models是配对的实验结果

 
 ## prepare data for baseline
 line.ground.truth = select(df_all,  project, train, test, filename, file.level.ground.truth, prediction.prob, line.number, line.level.ground.truth)
 line.ground.truth = filter(line.ground.truth, file.level.ground.truth == "True" & prediction.prob >= 0.5)
 line.ground.truth = distinct(line.ground.truth)
 
 get.line.metrics.result = function(baseline.df, cur.df.file) #glance.ea.result, cur.df.file
 {
   baseline.df.with.ground.truth = merge(baseline.df, cur.df.file, by=c("filename", "line.number"))
   #经过merge函数之后，line.number 现在指的是每个文件里面有bug的那一行

   ## 同一文件内的行为一组，按line.score从大到小降序排列；每一组内独立编号order
   sorted = baseline.df.with.ground.truth %>% group_by(filename) %>% arrange(-line.score, .by_group = TRUE) %>% mutate(order = row_number())
   
   ## IFA:  每个文件一个IFA，在每个文件的行组中取order最低的行号，代表检查到第一个有缺陷的行时需要检查多少行
   IFA = sorted %>% filter(line.level.ground.truth == "True") %>% group_by(filename)  %>% top_n(1, -order)
 
   ## added 2023-09-16 确保按文件名排序
   IFA = IFA%>% arrange(filename)
 
   ifa.list = IFA$order  #ifa.list里面就可以保存每个文件的ifa数值
   
   ##统计每个有缺陷的文件中包含多少个有缺陷的代码行
   # summarize()函数可以新建一个名叫total_true的列，这个列存储每个缺陷文件包含了多少个缺陷的代码行
   total_true = sorted %>%  group_by(filename) %>% summarize(total_true = sum(line.level.ground.truth == "True"))
 
    ##glance的预测结果中包含file.level.ground.truth == "FALSE"的文件，
    ##为使用DeepLineDP使用的性能指标，这些文件需要排除掉
   total_true = total_true %>% filter(total_true > 0)
 
   ## added 2023-09-16 确保按文件名排序
   total_true = total_true%>% arrange(filename)
   
   # Recall20%LOC: 按行line.score从大到小排好序后，计算前20%LOC对应的行级缺陷召回率
   # round()函数用于计算order/n之后保留2位小数点
   recall20LOC = sorted %>% group_by(filename) %>% mutate(effort = round(order/n(),digits = 2 )) %>% filter(effort <= 0.2) %>%
     summarize(correct_pred = sum(line.level.ground.truth == "True")) %>%
     merge(total_true) %>% mutate(recall20LOC = correct_pred/total_true)
 
   ## added 2023-09-16 确保按文件名排序
   recall20LOC = recall20LOC %>% arrange(filename)
   
   recall.list = recall20LOC$recall20LOC
 
   #Effort20%Recall：按行line.score从大到小排好序后，计算达到20%行级缺陷召回率时所需要审查的代码行占比，即effort
   # mutate()函数用于新建或者修改列
   effort20Recall = sorted %>% merge(total_true) %>% group_by(filename) %>% mutate(cummulative_correct_pred = cumsum(line.level.ground.truth == "True"), recall = round(cumsum(line.level.ground.truth == "True")/total_true, digits = 2)) %>%
     summarise(effort20Recall = sum(recall <= 0.2)/n())
 
   ## added 2023-09-16 确保按文件名排序
   effort20Recall = effort20Recall %>% arrange(filename)
   
   effort.list = effort20Recall$effort20Recall
 
   ## 按文件名排序后，才能保证每行是相同文件的ifa、recall和effort
   result.df = data.frame(ifa.list, recall.list, effort.list)
   
   return(result.df)
 }
 
 all_eval_releases = c('activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0', 
                       'camel-2.10.0', 'camel-2.11.0' , 
                       'derby-10.5.1.1' , 'groovy-1_6_BETA_2' , 'hbase-0.95.2', 
                       'hive-0.12.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1',  
                       'lucene-3.0.0', 'lucene-3.1', 'wicket-1.5.3')
 
 glance.ea.result.dir = 'D:/Gitee-code/DeepLineDP_2.0/output/prediction/GLANCE/GLANCE_EA/Line_Level/'
 
 glance.md.result.dir = 'D:/Gitee-code/DeepLineDP_2.0/output/prediction/GLANCE/GLANCE_EA/Line_Level/'
 # glance.rf.result.dir代表的就是GLANCE_LR
 glance.rf.result.dir = 'D:/Gitee-code/DeepLineDP_2.0/output/prediction/GLANCE/GLANCE_LR/Line_Level/'
 
 glance.ea.result.df = NULL
 glance.md.result.df = NULL
 glance.rf.result.df = NULL 
 
 ## get result from baseline
 for(rel in all_eval_releases)
 {  
  #filename是文件名，line_number是预测哪一行代码的有bug的行数，predicted_buggy_score是该行代码的bug得分
   glance.ea.result = read.csv(paste0(glance.ea.result.dir,rel,'-result.csv'))
   glance.ea.result$linenumber <- as.integer(sub('.*:', '', glance.ea.result$predicted_buggy_lines))
   glance.ea.result$filename <- sub(':.*', '', glance.ea.result$predicted_buggy_lines)
   glance.ea.result = select(glance.ea.result, "filename", "linenumber","predicted_buggy_score")
   names(glance.ea.result) = c("filename", "line.number", "line.score")
 
   glance.md.result = read.csv(paste0(glance.md.result.dir,rel,'-result.csv'))
   glance.md.result$linenumber <- as.integer(sub('.*:', '', glance.md.result$predicted_buggy_lines))
   glance.md.result$filename <- sub(':.*', '', glance.md.result$predicted_buggy_lines)
   glance.md.result = select(glance.md.result, "filename", "linenumber","predicted_buggy_score")
   names(glance.md.result) = c("filename", "line.number", "line.score")
  
   glance.rf.result = read.csv(paste0(glance.rf.result.dir,rel,'-result.csv'))
   glance.rf.result$linenumber <- as.integer(sub('.*:', '', glance.rf.result$predicted_buggy_lines))
   glance.rf.result$filename <- sub(':.*', '', glance.rf.result$predicted_buggy_lines)
   glance.rf.result = select(glance.rf.result, "filename", "linenumber","predicted_buggy_score")
   names(glance.rf.result) = c("filename", "line.number", "line.score")
 
   cur.df.file = filter(line.ground.truth, test==rel)
   cur.df.file = select(cur.df.file, filename, line.number, line.level.ground.truth)
 
   ##"%>% mutate(test=rel)" 确保记录下每个target project的名称
   glance.ea.eval.result = get.line.metrics.result(glance.ea.result, cur.df.file) %>% mutate(test=rel)
   glance.md.eval.result = get.line.metrics.result(glance.md.result, cur.df.file) %>% mutate(test=rel)
   glance.rf.eval.result = get.line.metrics.result(glance.rf.result, cur.df.file) %>% mutate(test=rel)
 
   glance.rf.result.df = rbind(glance.rf.result.df, glance.rf.eval.result)
   glance.ea.result.df = rbind(glance.ea.result.df, glance.ea.eval.result)
   glance.md.result.df = rbind(glance.md.result.df, glance.md.eval.result)
 
   print(paste0('finished ', rel))
 }
 
 
 #Force attention score of comment line is 0
 df_all[df_all$is.comment.line == "True",]$token.attention.score = 0
 
 tmp.top.k = get.top.k.tokens(df_all, 1500)
 
 merged_df_all = merge(df_all, tmp.top.k, by=c('project', 'train', 'test', 'filename', 'token'), all.x = TRUE)
 
 merged_df_all[is.na(merged_df_all$flag),]$token.attention.score = 0
 
 ## use top-k tokens 
 sum_line_attn = merged_df_all %>% filter(file.level.ground.truth == "True" & prediction.label == "True") %>% group_by(test, filename,is.comment.line, file.level.ground.truth, prediction.label, line.number, line.level.ground.truth) %>%
   summarize(attention_score = sum(token.attention.score), num_tokens = n())
 
 sorted = sum_line_attn %>% group_by(test, filename) %>% arrange(-attention_score, .by_group=TRUE) %>% mutate(order = row_number())
 
 ## get result from DeepLineDP
 # calculate IFA
 IFA = sorted %>% filter(line.level.ground.truth == "True") %>% group_by(test, filename) %>% top_n(1, -order)
 
 ## added 2023-09-16 确保按文件名排序
 IFA = IFA %>% arrange(test, filename)
 
 total_true = sorted %>% group_by(test, filename) %>% summarize(total_true = sum(line.level.ground.truth == "True"))
 
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
 deeplinedp.ifa = IFA$order
 deeplinedp.recall = recall20LOC$recall20LOC
 deeplinedp.effort = effort20Recall$effort20Recall
 
 deepline.dp.line.result = data.frame(deeplinedp.ifa, deeplinedp.recall, deeplinedp.effort, IFA$test)
 
 glance.rf.result.df = glance.rf.result.df %>%   summarize(IFA=median(ifa.list),recall=median(recall.list),effort=median(effort.list), .by=test)
 glance.ea.result.df = glance.ea.result.df %>%   summarize(IFA=median(ifa.list),recall=median(recall.list),effort=median(effort.list), .by=test)
 glance.md.result.df = glance.md.result.df %>%   summarize(IFA=median(ifa.list),recall=median(recall.list),effort=median(effort.list), .by=test)
 deepline.dp.line.result = deepline.dp.line.result %>%   summarize(IFA=median(deeplinedp.ifa),recall=median(deeplinedp.recall),effort=median(deeplinedp.effort), .by=IFA.test)
 
 names(glance.rf.result.df) = c("release", "IFA", "Recall20%LOC", "Effort@20%Recall")
 names(glance.ea.result.df) = c("release", "IFA", "Recall20%LOC", "Effort@20%Recall")
 names(glance.md.result.df)  = c("release", "IFA", "Recall20%LOC", "Effort@20%Recall")
 names(deepline.dp.line.result) = c("release", "IFA", "Recall20%LOC", "Effort@20%Recall")
 
 
 glance.rf.result.df$technique = 'GLANCERF'
 glance.ea.result.df$technique = 'GLANCEEA'
 glance.md.result.df$technique = 'GLANCEMD'
 deepline.dp.line.result$technique = 'DeepLineDP'
 
 
 all.line.result = rbind(glance.rf.result.df, glance.ea.result.df, glance.md.result.df, deepline.dp.line.result)
 
 recall.result.df = select(all.line.result, c('technique', 'Recall20%LOC'))
 ifa.result.df = select(all.line.result, c('technique', 'IFA'))
 effort.result.df = select(all.line.result, c('technique', 'Effort@20%Recall'))
 
 recall.result.df = preprocess(recall.result.df, FALSE)
 ifa.result.df = preprocess(ifa.result.df, TRUE)
 effort.result.df = preprocess(effort.result.df, TRUE)
 
 ggplot(recall.result.df, aes(x=reorder(variable, -value, FUN=median), y=value)) + geom_boxplot() + facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + ylab("Recall@Top20%LOC") + xlab("")
 ggsave(paste0(save.fig.dir,"file-Recall@Top20LOC.pdf"),width=4,height=2.5)
 
 ggplot(effort.result.df, aes(x=reorder(variable, value, FUN=median), y=value)) + geom_boxplot() + facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + ylab("Effort@Top20%Recall") + xlab("")
 ggsave(paste0(save.fig.dir,"file-Effort@Top20Recall.pdf"),width=4,height=2.5)
 
 ggplot(ifa.result.df, aes(x=reorder(variable, value, FUN=median), y=value)) + geom_boxplot()  + coord_cartesian(ylim=c(0,175)) + facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + ylab("IFA") + xlab("")
 ggsave(paste0(save.fig.dir, "file-IFA.pdf"),width=4,height=2.5)


# # ---------------- Code for RQ4 -----------------------#
# 
# ## get within-project result
# deepline.dp.result$project = c("activemq", "activemq", "activemq", "camel", "camel", "derby", "groovy", "hbase", "hive", "jruby", "jruby", "lucene", "lucene", "wicket")
# 
# file.level.by.project = deepline.dp.result %>% group_by(project) %>% summarise(mean.AUC = mean(all.auc), mean.MCC = mean(all.mcc), mean.bal.acc = mean(all.bal.acc))
# 
# names(file.level.by.project) = c("project", "AUC", "MCC", "Balance Accurracy")
# 
# IFA$project = str_replace(IFA$test, '-.*','')
# recall20LOC$project = str_replace(recall20LOC$test, '-.*','')
# recall20LOC$project = as.factor(recall20LOC$project)
# effort20Recall$project = str_replace(effort20Recall$test, '-.*','')
# 
# ifa.each.project = IFA %>% group_by(project) %>% summarise(mean.by.project = mean(order))
# recall.each.project = recall20LOC %>% group_by(project) %>% summarise(mean.by.project = mean(recall20LOC))
# effort.each.project = effort20Recall %>% group_by(project) %>% summarise(mean.by.project = mean(effort20Recall))
# 
# line.level.all.mean.by.project = data.frame(ifa.each.project$project, ifa.each.project$mean.by.project, recall.each.project$mean.by.project, effort.each.project$mean.by.project)
# 
# names(line.level.all.mean.by.project) = c("project", "IFA", "Recall20%LOC", "Effort@20%Recall")
# 
# 
# ## get cross-project result
# 
# prediction.dir = '../output/prediction/DeepLineDP/cross-release/'
# 
# projs = c('activemq', 'camel', 'derby', 'groovy', 'hbase', 'hive', 'jruby', 'lucene', 'wicket')
# 
# 
# get.line.level.metrics = function(df_all)
# {
#   #Force attention score of comment line is 0
#   df_all[df_all$is.comment.line == "True",]$token.attention.score = 0
# 
#   sum_line_attn = df_all %>% filter(file.level.ground.truth == "True" & prediction.label == "True") %>% group_by(filename,is.comment.line, file.level.ground.truth, prediction.label, line.number, line.level.ground.truth) %>%
#     summarize(attention_score = sum(token.attention.score), num_tokens = n())
#   sorted = sum_line_attn %>% group_by(filename) %>% arrange(-attention_score, .by_group=TRUE) %>% mutate(order = row_number())
#   
#   # calculate IFA
#   IFA = sorted %>% filter(line.level.ground.truth == "True") %>% group_by(filename) %>% top_n(1, -order)
#   total_true = sorted %>% group_by(filename) %>% summarize(total_true = sum(line.level.ground.truth == "True"))
#   
#   # calculate Recall20%LOC
#   recall20LOC = sorted %>% group_by(filename) %>% mutate(effort = round(order/n(),digits = 2 )) %>% filter(effort <= 0.2) %>%
#     summarize(correct_pred = sum(line.level.ground.truth == "True")) %>%
#     merge(total_true) %>% mutate(recall20LOC = correct_pred/total_true)
# 
#   # calculate Effort20%Recall
#   effort20Recall = sorted %>% merge(total_true) %>% group_by(filename) %>% mutate(cummulative_correct_pred = cumsum(line.level.ground.truth == "True"), recall = round(cumsum(line.level.ground.truth == "True")/total_true, digits = 2)) %>%
#     summarise(effort20Recall = sum(recall <= 0.2)/n())
#   
#   all.ifa = IFA$order
#   all.recall = recall20LOC$recall20LOC
#   all.effort = effort20Recall$effort20Recall
#   
#   result.df = data.frame(all.ifa, all.recall, all.effort)
#   
#   return(result.df)
# }
# 
# 
# all.line.result = NULL
# all.file.result = NULL
# 
# 
# for(p in projs)
# {
#   actual.pred.dir = paste0(prediction.dir,p,'/')
#   
#   all.files = list.files(actual.pred.dir)
#   
#   all.auc = c()
#   all.mcc = c()
#   all.bal.acc = c()
#   all.src.projs = c()
#   all.tar.projs = c()
#   
#   for(f in all.files)
#   {
#     df = read.csv(paste0(actual.pred.dir,f))
# 
#     f = str_replace(f,'.csv','')
#     f.split = unlist(strsplit(f,'-'))
#     target = tail(f.split,2)[1]
#     
#     df = as_tibble(df)
#     
#     df.file = select(df, c('train', 'test', 'filename', 'file.level.ground.truth', 'prediction.prob', 'prediction.label'))
#     
#     df.file = distinct(df.file)
# 
#     file.level.result = get.file.level.metrics(df.file)
# 
#     AUC = file.level.result[1]
#     MCC = file.level.result[2]
#     bal.acc = file.level.result[3]
# 
#     all.auc = append(all.auc, AUC)
#     all.mcc = append(all.mcc, MCC)
#     all.bal.acc = append(all.bal.acc, bal.acc)
#     
#     all.src.projs = append(all.src.projs, p)
#     all.tar.projs = append(all.tar.projs,target)
# 
#     tmp.top.k = get.top.k.tokens(df, 1500)
#     
#     merged_df_all = merge(df, tmp.top.k, by=c('project', 'train', 'test', 'filename', 'token'), all.x = TRUE)
#     
#     merged_df_all[is.na(merged_df_all$flag),]$token.attention.score = 0
#     
#     line.level.result = get.line.level.metrics(merged_df_all)
#     line.level.result$src = p
#     line.level.result$target = target
# 
#     all.line.result = rbind(all.line.result, line.level.result)
# 
#     print(paste0('finished ',f))
#     
#   }
#   
#   file.level.result = data.frame(all.auc,all.mcc,all.bal.acc)
#   file.level.result$src = p
#   file.level.result$target = all.tar.projs
#   
#   all.file.result = rbind(all.file.result, file.level.result)
# 
#   print(paste0('finished ',p))
# 
# }
# 
# final.file.level.result = all.file.result %>% group_by(target) %>% summarize(auc = mean(all.auc), balance_acc = mean(all.bal.acc), mcc = mean(all.mcc))
# 
# final.line.level.result = all.line.result %>% group_by(target) %>% summarize(recall = mean(all.recall), effort = mean(all.effort), ifa = mean(all.ifa))
# 
# 