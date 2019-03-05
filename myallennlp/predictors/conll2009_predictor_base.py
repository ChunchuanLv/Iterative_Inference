from typing import List, Dict

from overrides import overrides

from allennlp.common.util import JsonDict, sanitize, group_by_count
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor

import json
from allennlp.models.archival import Archive, load_archive
from allennlp.common.util import import_submodules
FIELDS_2009 = ["id", "form", "lemma", "plemma", "pos", "ppos", "feat", "pfeat", "head", "phead", "deprel", "pdeprel", "fillpred"]#, "pred"]

@Predictor.register("dependency_srl_base")
class Conll2009_PredictorBase(Predictor): #CoNLL2009-ST-evaluation-English  CoNLL2009-ST-English-development
    def __init__(self, model: Model, dataset_reader: DatasetReader,
                 output_file_path= None) -> None:
        super().__init__(model, dataset_reader)
        self.result = []
        self.crt_instance_id_start = 0
        #print(self._dataset_reader.type)

        if output_file_path is not None:
            self.set_files(output_file_path)

    def set_files(self,output_file_path=
                 "/afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-evaluation-English.predict"):

        self.conll_format_file_path = output_file_path

        self.file_out = open(self.conll_format_file_path, 'w+')

    def to_conll_format(self, instance, annotated_sentence, output):
        #    print("output_key", output.keys())

        sentence_len = len(annotated_sentence)

        EXCEPT_LIST = ['@@PADDING@@', '@@UNKNOWN@@', '_']
        pred_candidates = instance["predicate_candidates"].labels
        predicate_indexes = instance["predicate_indexes"].labels

        max_num_slots = len(pred_candidates)
        predicates = ["_"] * sentence_len

        sense_argmax = output["sense_argmax"]
        predicted_arc_tags = output["predicted_arc_tags"]

        for sense_idx, idx, pred_candidate in zip(sense_argmax, predicate_indexes, pred_candidates):
            if pred_candidate[sense_idx] not in EXCEPT_LIST:
                predicates[idx] = pred_candidate[sense_idx]
        for idx in range(sentence_len):
            word = annotated_sentence[idx]
            arg_slots = ['_'] * max_num_slots
            for y in range(max_num_slots):  # output["arc_tags"].shape[1]):
                if predicted_arc_tags[idx][y] != -1:
                    arg_slots[y] = self._model.vocab.get_index_to_token_vocabulary("tags")[
                        predicted_arc_tags[idx][y]]

            pred_label = predicates[idx]
            string = '\t'.join([word[type] for type in FIELDS_2009] + [pred_label] + arg_slots)
            self.file_out.write(string + '\n')
        self.file_out.write('\n')

        return

    @overrides
    def predict_instance(self, instances: Instance) -> JsonDict:

        return self.predict_batch_instance([instances])[0]

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
       # print ("last_instances", instances[-1])
       # print ("last_", self._dataset_reader.annotated_sentences[-1])

        outputs = self._model.forward_on_instances(instances)
        outputs = sanitize(outputs)
        for instance,annotated_sentence, output in zip(instances,self._dataset_reader.annotated_sentences[self.crt_instance_id_start:self.crt_instance_id_start+len(outputs)], outputs):
            #print(self.crt_instance_id_start, len(outputs))
            #print("output", output["tokens"])
            #assert False
            self.to_conll_format(instance,annotated_sentence, output)
        self.crt_instance_id_start += len(outputs)
        self.result += outputs
        return outputs