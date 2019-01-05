from typing import Dict, Tuple, List
import logging,os

from overrides import overrides
from conllu.parser import parse_line, DEFAULT_FIELDS

from allennlp.common.tqdm import Tqdm
from collections import OrderedDict, defaultdict

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField,AdjacencyField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN,DEFAULT_OOV_TOKEN
NEGATIVE_PRED = "_"

from myallennlp.dataset_readers.MultiCandidatesSequence import MultiCandidatesSequence

import xml.etree.ElementTree as ET

def folder_to_files_path(folder,ends =".txt"):
    files = os.listdir(folder)
    files_path = []
    for f in files:
        if f.endswith(ends):
            files_path.append(folder+f)
          #  break
    return   files_path

class PropbankReader:
    def parse(self):
        self.frames = dict()
        for f in self.frame_files_path:
            self.parse_file(f)

    def __init__(self, folder_path):
        self.frame_files_path = folder_to_files_path(folder_path +"/", ".xml")
        self.parse()

    def parse_file(self, f):
        tree = ET.parse(f)
        root = tree.getroot()
        for child in root:
            if child.tag == "predicate":
                self.add_lemma(child)

    # add cannonical amr lemma to possible set of words including for aliases of the words
    def add_lemma(self, node):
        lemma = node.attrib["lemma"].replace("_", "-")
        self.frames.setdefault(lemma, [])
        #    self.frames[lemma] = set()
        for child in node:
            if child.tag == "roleset":
                sensed_predicate = child.attrib["id"]
                self.frames[lemma].append(sensed_predicate)
                true_lemma =  sensed_predicate.split(".")[0]
                if sensed_predicate not in self.frames.setdefault(true_lemma, []):
                    self.frames[true_lemma].append(sensed_predicate)


    def get_frames(self):
        return self.frames

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

FIELDS_2009 = ["id", "form", "lemma", "plemma", "pos", "ppos", "feat", "pfeat", "head", "phead", "deprel", "pdeprel", "fillpred", "pred"]

import re

def parse_sentence(sentence_blob: str) -> Tuple[List[Dict[str, str]], List[Tuple[int, int]], List[str]]:
    """
    Parses a chunk of text in the SemEval SDP format.

    Each word in the sentence is returned as a dictionary with the following
    format:
    'id': '1',
    'form': 'Pierre',
    'lemma': 'Pierre',
    'pos': 'NNP',
    'head': '2',   # Note that this is the `syntactic` head.
    'deprel': 'nn',
    'top': '-',
    'pred': '+',
    'frame': 'named:x-c'

    Along with a list of arcs and their corresponding tags. Note that
    in semantic dependency parsing words can have more than one head
    (it is not a tree), meaning that the list of arcs and tags are
    not tied to the length of the sentence.
    """
    annotated_sentence = []
    arc_indices = []
    arc_tags = []
    predicates = []

    lines = [line.split("\t") for line in sentence_blob.split("\n")
             if line and not line.strip().startswith("#")]
    for line_idx, line in enumerate(lines):
        annotated_token = {k:v for k, v in zip(FIELDS_2009, line)}
        if annotated_token['fillpred'] == "Y":
            predicates.append(line_idx)
        annotated_sentence.append(annotated_token)

    for line_idx, line in enumerate(lines):
        for predicate_idx, arg in enumerate(line[len(FIELDS_2009):]):
            if arg != "_":
                arc_indices.append((line_idx, predicates[predicate_idx]))
                arc_tags.append(arg)
    return annotated_sentence, arc_indices, arc_tags,predicates


def lazy_parse(text: str):
    for sentence in text.split("\n\n"):
        if sentence:
            yield parse_sentence(sentence)



@DatasetReader.register("conll2009")
class Conll2009DatasetReader(DatasetReader):
    """
    Reads a file in the conllu Universal Dependencies format.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        The token indexers to be applied to the words TextField.
    use_language_specific_pos : ``bool``, optional (default = False)
        Whether to use UD POS tags, or to use the language specific POS tags
        provided in the conllu format.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 use_gold: bool = False,
                 lazy: bool = False,
                 data_folder = None) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.use_gold = use_gold
        self.lemma_to_sensed = self.read_frames(data_folder)

    def read_frames(self,data_folder):
        nbbank = PropbankReader(data_folder+"/nb_frames").get_frames()
        pbbank = PropbankReader(data_folder+"/pb_frames").get_frames()

        out = defaultdict(lambda:[],nbbank)
        for lemma in pbbank:
            if lemma in out:
                for pred in pbbank[lemma]:
                    if pred not in out[lemma]:
                        out[lemma].append(pred)
            else:
                out[lemma] = pbbank[lemma]
        return  out

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        training = "train" in file_path or "development" in file_path
        logger.info("Reading conll2009 srl data from: %s", file_path)

        with open(file_path) as sdp_file:
            for annotated_sentence, directed_arc_indices, arc_tags , predicates in lazy_parse(sdp_file.read()):
                # If there are no arc indices, skip this instance.
                if not directed_arc_indices:
                    continue
                tokens = [word["form"] for word in annotated_sentence]
                pred_candidates, pred_indexes, preds = self.data_for_sense_prediction(annotated_sentence,training)
                pos_tags = [word["pos"] for word in annotated_sentence] if self.use_gold else [word["ppos"] for word in annotated_sentence]
                verb_indicator = [1 if word["fillpred"] == "Y" else 0 for word in annotated_sentence]
                yield self.text_to_instance(tokens, verb_indicator,pos_tags, directed_arc_indices, arc_tags,pred_candidates, pred_indexes, preds)


    def data_for_sense_prediction(self,annotated_sentence,training):
        pred_candidates = []
        preds = [ word["pred"] if word["fillpred"] == "Y" else NEGATIVE_PRED for word in annotated_sentence]
        pred_indexes = []
        for word in annotated_sentence:
            if word["fillpred"] == "Y":
                pred = word["pred"]
                lemma = word["plemma"]
                if pred not in self.lemma_to_sensed[lemma] :
                    if training:
                        self.lemma_to_sensed[lemma].append(pred)
                        pred_indexes.append(self.lemma_to_sensed[lemma].index(pred ))
                        pred_candidates.append(self.lemma_to_sensed[lemma] )
                    else:
                        if  lemma in self.lemma_to_sensed:
                            pred_candidates.append(self.lemma_to_sensed[lemma] )
                            if pred in self.lemma_to_sensed[lemma]:
                                pred_indexes.append(self.lemma_to_sensed[lemma].index(pred ))
                            else:
                                pred_indexes.append(0)
                        elif lemma.endswith("s") and lemma[:-1] in self.lemma_to_sensed:
                            pred_candidates.append(self.lemma_to_sensed[lemma[:-1]] )
                            if pred in self.lemma_to_sensed[lemma]:
                                pred_indexes.append(self.lemma_to_sensed[lemma[:-1]].index(pred ))
                            else:
                                pred_indexes.append(0)
                        else:
                #            print (word["lemma"],word["plemma"],self.lemma_to_sensed[word["lemma"]],self.lemma_to_sensed[word["plemma"]],word["pred"])
                            pred_indexes.append(0)
                            pred_candidates.append([])
                else:
                    pred_indexes.append(self.lemma_to_sensed[lemma].index(pred ))
                    pred_candidates.append(self.lemma_to_sensed[lemma] )
            else:
                pred_indexes.append(0)
                pred_candidates.append([NEGATIVE_PRED])
        return pred_candidates,pred_indexes,preds
    @overrides
    def text_to_instance(self, # type: ignore
                         tokens: List[str],
                         verb_label: List[int],
                         pos_tags: List[str] = None,
                         arc_indices: List[Tuple[int, int]] = None,
                         arc_tags: List[str] = None,
                         pred_candidates:List[List[str]] = None,
                         pred_indexes:List[int] = None,
                         preds:List[str] = None,) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        token_field = TextField([Token(t) for t in tokens], self._token_indexers)
        fields["tokens"] = token_field
        fields['verb_indicator'] = SequenceLabelField(verb_label, token_field,label_namespace="indicator")
        fields["metadata"] = MetadataField({"tokens": tokens})
        if pos_tags is not None:
            fields["pos_tags"] = SequenceLabelField(pos_tags, token_field, label_namespace="pos")
        if arc_indices is not None and arc_tags is not None:
            fields["arc_tags"] = AdjacencyField(arc_indices, token_field, arc_tags,label_namespace="tags")

        fields["preds"] = SequenceLabelField(preds, token_field,label_namespace="pred")
        fields["pred_candidates"] = MultiCandidatesSequence(pred_candidates, token_field,label_namespace="pred")
        fields["pred_indexes"] = SequenceLabelField(pred_indexes, token_field,label_namespace="pred_indexes")

        return Instance(fields)




def main():
    reader = Conll2009DatasetReader(data_folder = "/afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p2/data/CoNLL2009-ST-English")
    print ("n sense",reader.max_sense)
    for i,b in enumerate(reader.lemma_to_sensed):
        print (b,reader.lemma_to_sensed[b])
        if i > 10: break
    train_data = reader._read("/afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-English-train.txt")


    print ("n sense",reader.get_max_sense())


    dev_data = reader._read("/afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-English-development.txt")
    for i,b in enumerate(dev_data):
        print (b)
        if i > 10: return


if __name__ == "__main__":
    main()
