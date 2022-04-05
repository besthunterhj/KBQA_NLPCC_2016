import time
from collections import Counter

import numpy as np
import torch
from text2vec import Word2Vec, SentenceModel, semantic_search
from transformers import BertTokenizer, BertForSequenceClassification, Trainer

from bert_for_ner_predict import predict_entity
from test_datasets import Dataset

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def cosine_similarity(x,y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    result = 0.0

    if denom != 0:
        result = num / denom

    return result


def get_kb_data(kb_path: str):
    triples = []
    with open(kb_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

        for line in lines:
            temp_triple = (line.strip().split("\t")[0], line.strip().split("\t")[1], line.strip().split("\t")[-1])
            triples.append(temp_triple)

    return triples


def get_candidate_attr_answer(input_question, ner_model, kb_path):
    # Load the data
    triples = get_kb_data(kb_path)

    # Get all entities from triples
    kb_entities = list(set([item[0] for item in triples]))

    # Get the Entity
    entity = predict_entity(input_question, ner_model)
    print(entity)
    # Figure out: Is the NER result in the kb_entities?
    if entity in kb_entities:

        # Get the candidate triples
        candidate_triple = []
        for item in triples:
            if entity == item[0]:
                candidate_triple.append(item)

        # Get the candidate answers
        candidate_attribute = []
        candidate_answer = []
        for item in candidate_triple:
            candidate_attribute.append(item[1])
            candidate_answer.append(item[-1])

        return candidate_attribute, candidate_answer, entity

    else:
        embedder = SentenceModel()
        entity_encoded = embedder.encode(entity)

        kb_entities_encoded = embedder.encode(kb_entities)
        hits = semantic_search(entity_encoded, kb_entities_encoded, top_k=1)[0]
        most_similar_entity_index = hits[0]['corpus_id']
        most_similar_entity = kb_entities[most_similar_entity_index]

        print("找不到关于 ", entity, " 的问题答案；", "您要问的是关于 ", most_similar_entity, "的问题吗？")
        feedback = "找不到关于 " + entity + " 的问题答案；" + "您要问的是关于 " + most_similar_entity + "的问题吗？"
        return [], [], feedback


def predict_candidate_answer(input_question, candidate_answers, qa_model_path):
    # Tokenizer only accept the list
    question_answer_list = []
    for item in candidate_answers:
        temp = [input_question, item]
        question_answer_list.append(temp)

    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained(qa_model_path)
    input_encode = tokenizer(question_answer_list, padding=True, truncation=True, max_length=512)

    # Build the dataset and model
    input_dataset = Dataset(input_encode, [0 for i in range(len(question_answer_list))])
    model = BertForSequenceClassification.from_pretrained(qa_model_path).to(device)

    # Predict
    trainer = Trainer(model=model)
    result = trainer.predict(input_dataset)
    prediction_labels = np.argmax(result.predictions, axis=1).tolist()
    y_pred = list(prediction_labels)

    return y_pred


def get_final_answer(input_question: str, candidate_answer: list, candidate_answer_status: list, candidate_attribute):
    # Get the number of "true" labels (from the model)
    count_true_label = Counter(candidate_answer_status)[1]

    # 如果 Bert 给候选答案进行分类后仍然有多个答案，则通过计算这些答案相对应的属性与问句的 similarity 来最终决定 final 答案
    if count_true_label > 1:

        w2vmodel = Word2Vec("w2v-light-tencent-chinese")
        encoded_question_encoded = w2vmodel.encode(input_question)

        true_label_answers_attr = []
        for i in range(len(candidate_answer_status)):
            if candidate_answer_status[i] == 1:
                pred_true_attr = candidate_attribute[i]
                true_label_answers_attr.append(pred_true_attr)

        # attrs_encoded = w2vmodel.encode(true_label_answers_attr)

        # similarities = cosine_similarity(encoded_question_encoded, attrs_encoded)
        # for item in true_label_answers_attr:
        #     embedding = w2vmodel.encode(item)
        #     embedding_vector = [item[0] for item in embedding.reshape(-1, 1)]
        #     similarity = cosine_similarity(np.array(encoded_question_vector), np.array(embedding_vector))
        #
        #     similarities.append(similarity)

        # most_similar_index = np.argmax(similarities, axis=0)
        # final_answer = candidate_answer[most_similar_index]

        embedder = SentenceModel()
        encoded_question_encoded = embedder.encode(input_question)

        # 获取由模型预测为真的答案所对应的属性值
        true_label_answers_attr = []
        for i in range(len(candidate_answer_status)):
            if candidate_answer_status[i] == 1:
                pred_true_attr = candidate_attribute[i]
                true_label_answers_attr.append(pred_true_attr)

        # 计算属性与问句的 similarity
        attribute_encoded = embedder.encode(true_label_answers_attr)
        hits = semantic_search(encoded_question_encoded, attribute_encoded, top_k=3)[0]

        # 在候选答案列表中检索最终答案
        final_attr_index = hits[0]['corpus_id']
        final_answer = candidate_answer[candidate_attribute.index(true_label_answers_attr[final_attr_index])]

    elif count_true_label == 1:
        final_answer = candidate_answer[candidate_answer_status.index(1)]

    else:
        final_answer = "NOT FOUND"
        # final_answer = "抱歉，知识库中暂未存有该问题的答案"

    return final_answer


def get_final_attr(input_question: str, candidate_answer_status: list, candidate_attribute):
    # Get the number of "true" labels (from the model)
    count_true_label = Counter(candidate_answer_status)[1]

    # 如果 Bert 给候选答案进行分类后仍然有多个答案，则通过计算这些答案相对应的属性与问句的 similarity 来最终决定 final 答案
    if count_true_label > 1:

        # w2vmodel = Word2Vec("w2v-light-tencent-chinese")
        # encoded_question_encoded = w2vmodel.encode(input_question)

        true_label_answers_attr = []
        for i in range(len(candidate_answer_status)):
            if candidate_answer_status[i] == 1:
                pred_true_attr = candidate_attribute[i]
                true_label_answers_attr.append(pred_true_attr)

        embedder = SentenceModel()
        encoded_question_encoded = embedder.encode(input_question)

        # 获取由模型预测为真的答案所对应的属性值
        true_label_answers_attr = []
        for i in range(len(candidate_answer_status)):
            if candidate_answer_status[i] == 1:
                pred_true_attr = candidate_attribute[i]
                true_label_answers_attr.append(pred_true_attr)

        # 计算属性与问句的 similarity
        attribute_encoded = embedder.encode(true_label_answers_attr)
        hits = semantic_search(encoded_question_encoded, attribute_encoded, top_k=1)[0]
        final_attr_index = hits[0]['corpus_id']
        final_attr = true_label_answers_attr[final_attr_index]

    elif count_true_label == 1:
        final_attr = candidate_attribute[candidate_answer_status.index(1)]

    else:
        final_attr = "UNKNOWN"

    return final_attr


def automated_qa(question, ner_model, qa_model, kb_path):
    # temp: to add the "space" between chars
    temp = ""
    for char in question:
        temp = temp + char + " "

    question = temp.strip()
    candidate_attribute, candidate_answers, entity = get_candidate_attr_answer(question, ner_model, kb_path)

    if candidate_answers and candidate_attribute:
        candidate_answer_status = predict_candidate_answer(question, candidate_answers, qa_model)
        final_answer = get_final_answer(question, candidate_answers, candidate_answer_status, candidate_attribute)

        return final_answer

    else:
        return entity


if __name__ == "__main__":
    start = time.time()
    answer = automated_qa("今天的推荐歌曲是什么？", "../model/bert_ner_model", "../model/bert_qa_model", "../data/test.kb")
    end = time.time()
    print(answer)
    interval = end - start
    print(interval)

    # input_question = input("请输入您的问题：")
    #
    # # temp: to add the "space" between chars
    # temp = ""
    # for char in input_question:
    #     temp = temp + char + " "
    #
    # input_question = temp.strip()
    #
    # candidate_attribute, candidate_answers, entity = get_candidate_attr_answer(input_question, "../model/bert_ner_model")
    #
    # if candidate_answers and candidate_attribute:
    #     candidate_answer_status = predict_candidate_answer(input_question, candidate_answers, "../model/bert_qa_model")
    #
    #     final_answer = get_final_answer(input_question, candidate_answers, candidate_answer_status, candidate_attribute)
    #
    #     print(final_answer)
