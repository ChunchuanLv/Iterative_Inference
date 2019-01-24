# Dependency_SRL 

After config srl2009.json, run

allennlp train exps/srl2009.json --include-package myallennlp --serialization-dir /afs/inf.ed.ac.uk/group/project/xchen13/dependency_srl_1_epoch

Train set:
allennlp predict /afs/inf.ed.ac.uk/group/project/xchen13/dependency_srl_1_epoch/model.tar.gz \
/afs/inf.ed.ac.uk/group/project/xchen13/datasets/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-English-train.txt \
--batch-size 128 --cuda-device 1 --use-dataset-reader --include-package myallennlp --predictor dependency_srl 


Test set:
allennlp predict /disk/scratch1/s1544871/srl/exps/score_0.1/model.tar.gz \
/afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-evaluation-English.txt \
--batch-size 100 --cuda-device 3 --use-dataset-reader --include-package myallennlp --predictor dependency_srl 


Dev set:
allennlp predict /afs/inf.ed.ac.uk/group/project/xchen13/tmp/model.tar.gz \
/afs/inf.ed.ac.uk/group/project/xchen13/datasets/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-English-development.txt \
--batch-size 128 --use-dataset-reader --include-package myallennlp --predictor dependency_srl 




# For evaluation:

perl eval09.pl -g /afs/inf.ed.ac.uk/group/project/xchen13/datasets/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-evaluation-English.txt -s /afs/inf.ed.ac.uk/group/project/xchen13/datasets/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-evaluation-English.txt

REF:
https://ufal.mff.cuni.cz/conll2009-st/scorer.html
   [perl] eval09.pl [OPTIONS] -g <gold standard> -s <system output>

TEST SET:
perl eval09.pl -g /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-evaluation-English.txt  \
 -s /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-evaluation-English.predict0 -q


DEV SET:
perl eval09.pl -g /afs/inf.ed.ac.uk/group/project/xchen13/datasets/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-English-development.txt_filtered \
 -s /afs/inf.ed.ac.uk/group/project/xchen13/tmp/CoNLL2009-ST-English-development.predict