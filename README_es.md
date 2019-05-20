# Dependency_SRL 

After config srl2009.json, run

allennlp train /disk/scratch1/s1544871/srl/exps/direct_soft_cs/config.json --include-package myallennlp --serialization-dir  /disk/scratch1/s1544871/srl/exps/direct_soft_cs -f
Train set:
allennlp predict  /disk/scratch1/s1544871/srl/exps/base_es/model.tar.gz \
/afs/inf.ed.ac.uk/group/project/xchen13/datasets/2009_conll_p1/data/CoNLL2009-ST-Spanish/CoNLL2009-ST-Spanish-train.txt \
--batch-size 128 --cuda-device 1 --use-dataset-reader --include-package myallennlp --predictor dependency_srl 


Test set:
longjob -28day -c "allennlp predict /disk/scratch1/s1544871/srl/exps/direct_cs_tie_2/model.tar.gz    /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Czech/CoNLL2009-ST-evaluation-Czech.txt --batch-size 64   --use-dataset-reader --include-package myallennlp --predictor dependency_srl"

allennlp predict  /disk/scratch1/s1544871/srl/exps/direct_de_clean_2/model.tar.gz  \
/afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-German/CoNLL2009-ST-evaluation-German.txt \
--batch-size 200   --use-dataset-reader --include-package myallennlp --predictor dependency_srl

allennlp predict  /disk/scratch1/s1544871/srl/exps/direct_jp_auto_2/model.tar.gz \
/afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Japanese/CoNLL2009-ST-evaluation-Japanese.txt \
--batch-size 128  --use-dataset-reader --include-package myallennlp --predictor dependency_srl



allennlp predict  /disk/scratch1/s1544871/srl/exps/direct_ca_auto_2/model.tar.gz \
/afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Catalan/CoNLL2009-ST-evaluation-Catalan.txt \
--batch-size 128   --use-dataset-reader --include-package myallennlp --predictor dependency_srl

allennlp predict  /disk/scratch/s1544871/srl/exps/direct_es_auto_2/model.tar.gz  \
/afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Spanish/CoNLL2009-ST-evaluation-Spanish.txt \
--batch-size 128   --use-dataset-reader --include-package myallennlp --predictor dependency_srl

Dev set:
allennlp predict  /disk/scratch1/s1544871/srl/exps/direct_ca_auto_2/model.tar.gz \
/afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Catalan/CoNLL2009-ST-Catalan-development.txt \
--batch-size 128 --use-dataset-reader  --include-package myallennlp --predictor dependency_srl

allennlp predict  /disk/scratch/s1544871/srl/exps/direct_es_auto_2/model.tar.gz  \
/afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Spanish/CoNLL2009-ST-Spanish-development.txt \
--batch-size 128 --use-dataset-reader  --include-package myallennlp --predictor dependency_srl




allennlp predict  /disk/scratch1/s1544871/srl/exps/direct_jp_auto_2/model.tar.gz \
/afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Japanese/CoNLL2009-ST-Japanese-development.txt \
--batch-size 128 --use-dataset-reader  --include-package myallennlp --predictor dependency_srl

allennlp predict  /disk/scratch1/s1544871/srl/exps/direct_de_auto_2/model.tar.gz /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-German/CoNLL2009-ST-German-development.txt --batch-size 128 --use-dataset-reader  --include-package myallennlp --predictor dependency_srl


longjob -28day -c "allennlp predict  /disk/scratch1/s1544871/srl/exps/direct_cs_auto_2/model.tar.gz /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Czech/CoNLL2009-ST-Czech-development.txt --batch-size 64   --use-dataset-reader --include-package myallennlp --predictor dependency_srl"



OOD set:
longjob -28day -c "allennlp predict  /disk/scratch1/s1544871/srl/exps/direct_cs_clean_2/model.tar.gz  /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Czech/CoNLL2009-ST-evaluation-Czech-ood.txt --batch-size 64   --use-dataset-reader --include-package myallennlp --predictor dependency_srl"


allennlp predict /disk/scratch1/s1544871/srl/exps/spen_soft_cs/model.tar.gz \
/afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Czech/CoNLL2009-ST-evaluation-Czech-ood.txt \
--batch-size 128   --cuda-device 2 --use-dataset-reader --include-package myallennlp --predictor dependency_srl 

allennlp predict  /disk/scratch1/s1544871/srl/exps/direct_de_tie/model.tar.gz  \
/afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-German/CoNLL2009-ST-evaluation-German-ood.txt \
--batch-size 128 --use-dataset-reader --cuda-device 2 --include-package myallennlp --predictor dependency_srl 


Evaluate

allennlp predict  /disk/scratch_big/s1544871/srl/exps/base_es/model.tar.gz \
/afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Spanish/CoNLL2009-ST-evaluation-Spanish.txt \
 --cuda-device 1 --include-package myallennlp

# For evaluation:

perl eval09.pl -g /afs/inf.ed.ac.uk/group/project/xchen13/datasets/2009_conll_p1/data/CoNLL2009-ST-Spanish/CoNLL2009-ST-evaluation-Spanish.txt -s /afs/inf.ed.ac.uk/group/project/xchen13/datasets/2009_conll_p1/data/CoNLL2009-ST-Spanish/CoNLL2009-ST-evaluation-Spanish.txt

REF:
https://ufal.mff.cuni.cz/conll2009-st/scorer.html
   [perl] eval09.pl [OPTIONS] -g <gold standard> -s <system output>

TEST SET:
perl eval09.pl -g /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-German/CoNLL2009-ST-evaluation-German.txt  \
 -s /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-German/CoNLL2009-ST-evaluation-German.predict -q
perl eval09.pl -g /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-German/CoNLL2009-ST-evaluation-German.txt  \
 -s /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-German/CoNLL2009-ST-evaluation-German.predict2 -q
 
perl eval09.pl -g /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Japanese/CoNLL2009-ST-evaluation-Japanese.txt  \
 -s /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Japanese/CoNLL2009-ST-evaluation-Japanese.predict -q
perl eval09.pl -g /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Japanese/CoNLL2009-ST-evaluation-Japanese.txt  \
 -s /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Japanese/CoNLL2009-ST-evaluation-Japanese.predict2 -q
 
perl eval09.pl -g /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Catalan/CoNLL2009-ST-evaluation-Catalan.txt  \
 -s /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Catalan/CoNLL2009-ST-evaluation-Catalan.predict2 -q

perl eval09.pl -g /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Czech/CoNLL2009-ST-evaluation-Czech.txt  \
 -s /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Czech/CoNLL2009-ST-evaluation-Czech.predict -q
perl eval09.pl -g /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Czech/CoNLL2009-ST-evaluation-Czech.txt  \
 -s /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Czech/CoNLL2009-ST-evaluation-Czech.predict2 -q
 
perl eval09.pl -g /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Spanish/CoNLL2009-ST-evaluation-Spanish.txt  \
 -s /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Spanish/CoNLL2009-ST-evaluation-Spanish.predict2 -q


DEV SET:
perl eval09.pl -g /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-German/CoNLL2009-ST-German-development.txt \
 -s /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-German/CoNLL2009-ST-German-development.predict -q
perl eval09.pl -g /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-German/CoNLL2009-ST-German-development.txt \
 -s /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-German/CoNLL2009-ST-German-development.predict2 -q
 
 
perl eval09.pl -g /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Japanese/CoNLL2009-ST-Japanese-development.txt \
 -s /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Japanese/CoNLL2009-ST-Japanese-development.predict -q
perl eval09.pl -g /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Japanese/CoNLL2009-ST-Japanese-development.txt \
 -s /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Japanese/CoNLL2009-ST-Japanese-development.predict2 -q
 
perl eval09.pl -g /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Czech/CoNLL2009-ST-Czech-development.txt \
 -s /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Czech/CoNLL2009-ST-Czech-development.predict -q
perl eval09.pl -g /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Czech/CoNLL2009-ST-Czech-development.txt \
 -s /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Czech/CoNLL2009-ST-Czech-development.predict2 -q
 
perl eval09.pl -g /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Catalan/CoNLL2009-ST-Catalan-development.txt \
 -s /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Catalan/CoNLL2009-ST-Catalan-development.predict -q
perl eval09.pl -g /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Catalan/CoNLL2009-ST-Catalan-development.txt \
 -s /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Catalan/CoNLL2009-ST-Catalan-development.predict2 -q
 
perl eval09.pl -g /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Spanish/CoNLL2009-ST-Spanish-development.txt \
 -s /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Spanish/CoNLL2009-ST-Spanish-development.predict -q
perl eval09.pl -g /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Spanish/CoNLL2009-ST-Spanish-development.txt \
 -s /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Spanish/CoNLL2009-ST-Spanish-development.predict2 -q
 
 
 
 OOD
 
perl eval09.pl -g /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-German/CoNLL2009-ST-evaluation-German-ood.txt  \
 -s /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-German/CoNLL2009-ST-evaluation-German-ood.predict2 -q
 
 
perl eval09.pl -g /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Czech/CoNLL2009-ST-evaluation-Czech-ood.txt  \
 -s /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Czech/CoNLL2009-ST-evaluation-Czech-ood.predict -q
perl eval09.pl -g /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Czech/CoNLL2009-ST-evaluation-Czech-ood.txt  \
 -s /afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Czech/CoNLL2009-ST-evaluation-Czech-ood.predict2 -q
 
 
 
 
 
allennlp predict /disk/scratch_big/s1544871/srl/exps/base_es/model.tar.gz \
/afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p1/data/CoNLL2009-ST-Spanish/CoNLL2009-ST-Spanish-development.txt \
--batch-size 100 --use-dataset-reader --cuda-device 1 --include-package myallennlp --predictor dependency_srl 