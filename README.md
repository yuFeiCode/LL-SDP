# The replication kit of Experiment
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
**DeepLineDP** contains the following directory

*  output: This directory contains the following sub-directories:
  
    *  `loss`: This directory stores training and validation loss
    *  `model`: This directory stores trained models
    *  `prediction`: This directory stores prediction (in CSV files) obtained from the trained models
    *  `Word2Vec_mode`l: This directory stores word2vec models of each software project

*  script: This directory contains the following directories and files:
    *  `preprocess_data.py`: The source code used to preprocess datasets for file-level model training and evaluation
    *  `export_data_for_line_level_baseline.py`: The source code used to prepare data for line-level baseline
    *  `my_util.py`: The source code used to store utility functions
    *  `train_word2vec.py`: The source code used to train word2vec models
    *  `DeepLineDP_model.py`: The source code that stores DeepLineDP architecture
    *  `train_model.py`: The source code used to train DeepLineDP models
    *  `generate_prediction.py`: The source code used to generate prediction (for RQ1-RQ3)

**GLANCE** contains the following directory

*  `src`: This folder stores the source code of GLANCE written in Python.

## Environment Setup

### Python Environment Setup

  1.clone the github respository by using the following command:
  
    git clone https://github.com/yuFeiCode/LL-SDP.git
    
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

    python main.py [model_name]

In above command,

*  [model_name] indicates a CLBI approach, GLANCE_EA, GLANCE_MD, and GLANCE_LR.
Here is some usage examples:

    `python main.py GLANCE_EA`

#### **As for DeepLineDP**

1.run the command to prepare data for file-level model training. The output will be stored in`./datasets/preprocessed_data`

    python preprocess_data.py

2.to train Word2Vec models, run the following command:

    python train_word2vec.py <DATASET_NAME>
    
Where <DATASET_NAME> is one of the following: `activemq`, `camel`, `derby`, `groovy`, `hbase`, `hive`, `jruby`, `lucene, wicket`


3.to train DeepLineDP models, run the following command:

    python train_model.py -dataset <DATASET_NAME>
    
The trained models will be saved in `./output/model/DeepLineDP/<DATASET_NAME>/`, and the loss will be saved in `../output/loss/DeepLineDP/<DATASET_NAME>-loss_record.csv`

4.to make a prediction of each software release, run the following command:

    python generate_prediction.py -dataset <DATASET_NAME>
    
The generated output is a csv file which contains the following information:

*  `project`: A software project, as specified by <DATASET_NAME>
*  `train`: A software release that is used to train DeepLineDP models
*  `test`: A software release that is used to make a prediction
*  `filename`: A file name of source code
*  `file-level-ground-truth`: A label indicating whether source code is clean or defective
*  `prediction-prob`: A probability of being a defective file
*  `prediction-label`: A prediction indicating whether source code is clean or defective
*  `line-number`: A line number of a source code file
*  `line-level-ground-truth`: A label indicating whether the line is modified
*  `is-comment-line`: A flag indicating whether the line is comment
*  `token`: A token in a code line
*  `token-attention-score`: An attention score of a token
*  `line-attebtion-score`: An attention score of a line
  
The generated output is stored in `./output/prediction/DeepLineDP/within-release/`
  
## Obtaining the Evaluation Result

Run `get_evaluation_result.R` to get the result of RQ1 and RQ2 (may run in IDE or by the following command)

  `Rscript  get_evaluation_result.R`

the result are figures that are sorted in `./figures`

## Contact us

**Mail**: zyf@stu.gxnu.edu.cn
