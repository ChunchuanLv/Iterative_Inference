from typing import Iterator, List, Dict
import torch
import torch.optim as optim
import numpy as np
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor
from allennlp.data.dataset_readers.universal_dependencies import UniversalDependenciesDatasetReader
from allennlp.models.biaffine_dependency_parser import BiaffineDependencyParser

torch.manual_seed(1)

reader = UniversalDependenciesDatasetReader()
train_dataset = reader.read("/afs/inf.ed.ac.uk/user/s15/s1544871/Data/Universal Dependencies 2.2/ud-treebanks-v2.2/UD_English-LinES/en_lines-ud-train.conllu")
validation_dataset = reader.read("/afs/inf.ed.ac.uk/user/s15/s1544871/Data/Universal Dependencies 2.2/ud-treebanks-v2.2/UD_English-LinES/en_lines-ud-dev.conllu")

instances = list(train_dataset)

print ("instances[0]",instances[0])
vocab = Vocabulary.from_instances(train_dataset + validation_dataset)




EMBEDDING_DIM = 6
HIDDEN_DIM = 6
tag_representation_dim = 6
arc_representation_dim = 6
token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('words'),
                            embedding_dim=EMBEDDING_DIM)
word_embeddings = BasicTextFieldEmbedder({"words": token_embedding})
lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
model = BiaffineDependencyParser(vocab,word_embeddings, lstm, vocab,tag_representation_dim,arc_representation_dim,dropout = 0.1)


optimizer = optim.SGD(model.parameters(), lr=0.1)
iterator = BucketIterator(batch_size=2, sorting_keys=[("sentence", "num_tokens")])
iterator.index_with(vocab)
trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=validation_dataset,
                  patience=10,
                  num_epochs=1000)
trainer.train()
predictor = SentenceTaggerPredictor(model, dataset_reader=reader)
tag_logits = predictor.predict("The dog ate the apple")['tag_logits']
tag_ids = np.argmax(tag_logits, axis=-1)
print([model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])