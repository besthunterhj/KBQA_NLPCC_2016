from transformers import BertForTokenClassification, BertForSequenceClassification, BertTokenizer
from text2vec import SentenceModel

NER_TOKENIZER = BertTokenizer.from_pretrained("../../model/bert_ner_model")
NER_MODEL = BertForTokenClassification.from_pretrained("../../model/bert_ner_model")

QA_TOKENIZER = BertTokenizer.from_pretrained("../../model/bert_qa_model")
QA_MODEL = BertForSequenceClassification.from_pretrained("../../model/bert_qa_model")

SENTENCEMODEL = SentenceModel()
