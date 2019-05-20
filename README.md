# Dependency_SRL 

After config srl2009.json, run

allennlp train exps/srl2009.json --include-package myallennlp --serialization-dir  /disk/scratch1/s1544871/srl/exps/refine

Train set:
allennlp predict /afs/inf.ed.ac.uk/group/project/xchen13/dependency_srl_1_epoch/model.tar.gz \
/afs/inf.ed.ac.uk/group/project/xchen13/datasets/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-English-train.txt \
--batch-size 128 --cuda-device 1 --use-dataset-reader --include-package myallennlp --predictor dependency_srl 


Dev set:

allennlp predict   /disk/scratch1/s1544871/srl/exps/direct_zh_auto_2/model.tar.gz \
/afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p2/data/CoNLL2009-ST-Chinese/CoNLL2009-ST-Chinese-development.txt \
--batch-size 128  --use-dataset-reader --include-package myallennlp --predictor dependency_srl 



allennlp predict /disk/scratch1/s1544871/srl/exps/direct_en_tie_2/model.tar.gz \
/afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-English-development.txt \
--batch-size 128   --use-dataset-reader --include-package myallennlp --predictor dependency_srl 

Test set:
allennlp predict /disk/scratch1/s1544871/srl/exps/direct_en_tie_2/model.tar.gz \
/afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-evaluation-English.txt \
--batch-size 128   --cuda-device 1 --use-dataset-reader --include-package myallennlp --predictor dependency_srl 



allennlp predict  /disk/scratch1/s1544871/srl/exps/direct_zh_tie_2/model.tar.gz  \
/afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p2/data/CoNLL2009-ST-Chinese/CoNLL2009-ST-evaluation-Chinese.txt \
--batch-size 128 --cuda-device 1   --use-dataset-reader --include-package myallennlp --predictor dependency_srl 



OOD set:
allennlp predict   /disk/scratch1/s1544871/srl/exps/direct_en_clean_2/model.tar.gz \
/afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-evaluation-English-ood.txt \
--batch-size 128 --use-dataset-reader --include-package myallennlp --predictor dependency_srl 



allennlp predict /afs/inf.ed.ac.uk/user/s15/s1544871/PycharmProjects/Iterative_Inference/base_model.tar.gz \
/afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-evaluation-English-ood.txt \
--batch-size 100 --cuda-device 3 --use-dataset-reader --include-package myallennlp --predictor dependency_srl 


Evaluate

allennlp evaluate  /disk/scratch/s1544871/srl/exps/no_gate_soft/model.tar.gz \
/afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-evaluation-English.txt \
 --cuda-device 3 --include-package myallennlp

# For evaluation:

perl eval09.pl -g /afs/inf.ed.ac.uk/group/project/xchen13/datasets/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-evaluation-English.txt -s /afs/inf.ed.ac.uk/group/project/xchen13/datasets/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-evaluation-English.txt

REF:
https://ufal.mff.cuni.cz/conll2009-st/scorer.html
   [perl] eval09.pl [OPTIONS] -g <gold standard> -s <system output>

TEST SET:
perl eval09.pl -g /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-evaluation-English.txt  \
 -s /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-evaluation-English.predict -q
perl eval09.pl -g /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-evaluation-English.txt  \
 -s /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-evaluation-English.predict1 -q
 
perl eval09.pl -g /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p2/data/CoNLL2009-ST-Chinese/CoNLL2009-ST-evaluation-Chinese.txt  \
 -s /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p2/data/CoNLL2009-ST-Chinese/CoNLL2009-ST-evaluation-Chinese.predict -q
perl eval09.pl -g /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p2/data/CoNLL2009-ST-Chinese/CoNLL2009-ST-evaluation-Chinese.txt  \
 -s /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p2/data/CoNLL2009-ST-Chinese/CoNLL2009-ST-evaluation-Chinese.predict2 -q

 

DEV SET:
perl eval09.pl -g /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-English-development.txt \
 -s /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-English-development.predict -q 
perl eval09.pl -g /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-English-development.txt \
 -s /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-English-development.predict2 -q 

 
perl eval09.pl -g /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p2/data/CoNLL2009-ST-Chinese/CoNLL2009-ST-Chinese-development.txt \
 -s /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p2/data/CoNLL2009-ST-Chinese/CoNLL2009-ST-Chinese-development.predict -q 
perl eval09.pl -g /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p2/data/CoNLL2009-ST-Chinese/CoNLL2009-ST-Chinese-development.txt \
 -s /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p2/data/CoNLL2009-ST-Chinese/CoNLL2009-ST-Chinese-development.predict2 -q 


ood SET
perl eval09.pl -g /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-evaluation-English-ood.txt  \
 -s /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-evaluation-English-ood.predict0 -q
 

 
allennlp predict /disk/scratch_big/s1544871/srl/exps/base_es/model.tar.gz \
/afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Spanish/CoNLL2009-ST-English-development.txt \
--batch-size 100 --use-dataset-reader --cuda-device 1 --include-package myallennlp --predictor dependency_srl 



perl eval09.pl -g /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p2/data/english_records_best/CoNLL2009-ST-evaluation-English.txt  \
 -s /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p2/data/english_records_best/CoNLL2009-ST-evaluation-English.predict0 -q
 