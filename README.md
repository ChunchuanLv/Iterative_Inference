# Semantic Role Labeling with Iterative Structure Refinement

This repository contains code for training semantic role labeling model described in:
[Semantic Role Labeling with Iterative Structure Refinement](https://www.aclweb.org/anthology/D19-1099.pdf)
We use English as example, other languages have similar configuration.

If you use our code, please cite our paper as follows:  
  > @inproceedings{lyu-etal-2019-semantic,  
  > &nbsp; &nbsp; title={Semantic Role Labeling with Iterative Structure Refinement},  
  > &nbsp; &nbsp; author={Lyu, Chunchuan  and
      Cohen, Shay B.  and
      Titov, Ivan},  
  > &nbsp; &nbsp; booktitle={"Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)"},  
  > &nbsp; &nbsp; month={nov}  
  > &nbsp; &nbsp; year={2019}  
  > }  

## Prerequisites:
* Python 3.6 
* pytorch 1.0
* 2009 CoNLL Shared Task

## Training:
Train baseline model
`allennlp train exps/srl2009_base.json --include-package myallennlp --serialization-dir  ../Iterative_Inference_Models/en_base`
Move out model file and vocabulary file for building refiner:
`cp ../Iterative_Inference_Models/en_base/model.tar.gz ../Iterative_Inference_Models/base_en_model.tar.gz`
`cp -r ../Iterative_Inference_Models/en_base/vocabulary ../Iterative_Inference_Models/en_vocabulary`
Train refiner
`allennlp train exps/srl2009.json --include-package myallennlp --serialization-dir  ../Iterative_Inference_Models/refine`

## Testing
Annotate 
`allennlp predict  ../Iterative_Inference_Models/refine/model.tar.gz [../CoNLL2009-ST-evaluation-English.txt] --batch-size 128   --cuda-device 0 --use-dataset-reader --include-package myallennlp --predictor dependency_srl`

## Evaluation

Official transcript:
https://ufal.mff.cuni.cz/conll2009-st/scorer.html
   [perl] eval09.pl [OPTIONS] -g <gold standard> -s <system output>

`perl eval09.pl -g [../CoNLL2009-ST-evaluation-English.txt] -s ../Iterative_Inference_Models/refine/CoNLL2009-ST-evaluation-English.predict -q`


## Contact
Contact (chunchuan.lv@gmail.com) if you have any questions!

 -s /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p2/data/english_records_best/CoNLL2009-ST-evaluation-English.predict0 -q
 