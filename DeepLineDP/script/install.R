# curl openssl systemfonts ModelMetrics TTR httr quantmod gargle googledrive googlesheets4 httr ragg readxl rvest xml2
install.packages(c("tidyverse", "gridExtra", "ModelMetrics", "caret", "reshape2", "pROC", "effsize", "ScottKnottESD"),repos = "https://mirrors.ustc.edu.cn/CRAN/")
# install.packages("ScottKnottESD", dependencies=TRUE)

# check your package library path 
# .libPaths("/usr/local/lib/R/site-library")

# # grab old packages names
# old_packages <- installed.packages(lib.loc = "/Library/Frameworks/R.framework/Versions/3.6/Resources/library")
# old_packages <- as.data.frame(old_packages)
# list.of.packages <- unlist(old_packages$Package)

# # remove old packages 
# remove.packages( installed.packages( priority = "NA" )[,1] )

# # reinstall all packages 
# new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
# if(length(new.packages)) install.packages(new.packages)
# lapply(list.of.packages,function(x){library(x,character.only=TRUE)})