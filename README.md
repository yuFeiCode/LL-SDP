# The replication kit of Experiment
##  Title: Deep line-level defect prediction: how far are we?
This repository stores the source codes of DeepLineDP and GLANCE.
## Datasets
The datasets are obtained from Wattanakriengkrai et. al. The datasets contain 32 software releases across 9 software projects. The [datasets](https://github.com/awsm-research/line-level-defect-prediction) that we used in our experiment can be found in this github .

The file-level datasets (in the File-level directory) contain the following columns

*  `File`: A file name of source code
*  `Bug`: A label indicating whether source code is clean or defective
*  `SRC`: A content in source code file

The line-level datasets (in the Line-level directory) contain the following columns

*  `File`: A file name of source code
*  `Line_number`: A line number where source code is modified
*  `SRC`: An actual source code that is modified

## Floders Introduction

## Environment Setup

### Python Environment Setup

  1.clone the github respository by using the following command:
  
    git clone https://github.com/yuFeiCode/How-far-are-we-.git
    
  2.download the datasets from the this [github](https://github.com/awsm-research/line-level-defect-prediction) and keep it in `DeepLineDP/datasets/original` and `GLANCE/Dataste/`
  
  3.use the following command to install required libraries in conda environment
  
    conda env create -f requirements.yml
          
    conda activate DeepLineDP_env

  4.install PyTorch library by following the instruction from this link (the installation instruction may vary based on OS and CUDA version)

### R Environment Setup

  Download the following package: `tidyverse`, `gridExtra`, `ModelMetrics`, `caret`, `reshape2`, `pROC`, `effsize`, `ScottKnottESD`
  
## Execution commands
### **As for GLANCE**
In order to make it easier to obtain the classification results, you can enter the GLANCE folder and run it according to the following command regulation.
`python main.py [model_name]`

In above command,

*  [model_name] indicates a CLBI approach, GLANCE_EA, GLANCE_MD, and GLANCE_LR.
Here is some usage examples:

`python main.py GLANCE_EA`

#### **As for DeepLineDP**
you can follow the project in this [github](https://github.com/awsm-research/DeepLineDP) repository to get the output file 

## Obtaining the Evaluation Result
Run `get_evaluation_result.R` to get the result of RQ1-RQ4 (may run in IDE or by the following command)

  `Rscript  get_evaluation_result.R`

the result are figures that are sorted in `./figures`
