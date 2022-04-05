

import torch
from text2vec import Word2Vec, SentenceModel, semantic_search
import numpy as np
from sklearn.metrics import classification_report

from bert_for_ner_predict import predict_entities
from test_datasets import Dataset
from kbqa_application import predict_candidate_answer, get_final_answer


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def cosine_similarity(x,y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    result = 0.0

    if denom != 0:
        result = num / denom

    return result


def extract_three_type_data(dataset: str) -> dict:

    question_label = "<question"
    triple_label = "<triple"
    answer_label = "<answer"

    question_list = []
    triple_list = []
    answer_list = []
    with open(dataset, "r", encoding="utf-8") as file:
        lines = file.readlines()

        for line in lines:
            expected_text = line.split("> ", 1)[-1].strip()
            if question_label in line:
                question_list.append(expected_text)
            elif triple_label in line:
                current_entity = expected_text.split("|||")[0].strip()
                current_attr = expected_text.split("|||")[1].strip()
                current_answ = expected_text.split("|||")[-1].strip()

                if " " in current_attr:
                    current_attr = current_attr.replace(" ", "")

                triple_list.append((current_entity, current_attr, current_answ))
            elif answer_label in line:
                answer_list.append(expected_text)
            else:
                continue

    return {
        "questions": question_list,
        "triples": triple_list,
        "answers": answer_list,
    }


def get_candidate_attr_answer(data_dict):
    # Load the data.
    triples = data_dict["triples"] # remember to sub the [:5]

    # Get all entities from triples
    kb_entities = list(set([item[0] for item in triples]))

    # Get the Entity
    entities = predict_entities(data_dict["questions"], "../bert_ner_model") # remember to sub the [:5]

    # Figure out: Is the NER result in the kb_entities?
    # question_dict Be like:
    # {
    #                 q1: {
    #                     entity: "",
    #                     ca_attr: [],
    #                     ca_answ: [],
    #                 },
    #                 q2: {...}
    # }
    question_dict = {}
    w2v_model = Word2Vec("w2v-light-tencent-chinese")
    for i in range(len(entities)):

        if entities[i] not in kb_entities:
            entity_encoded = w2v_model.encode(entities[i])
            entities_embedding = w2v_model.encode(kb_entities)

            similarity = cosine_similarity(entity_encoded, entities_embedding)

            most_similar_index = np.argmax(similarity, axis=0)
            most_similar_entity = kb_entities[most_similar_index]

            # Sub the not found entity
            entities[i] = most_similar_entity
            
        # Get the candidate triples
        print(entities[i])
        current_question_dict = {"entity": entities[i], }
        current_question_ca_triple = []
        for item in triples:
            if entities[i] == item[0]:
                current_question_ca_triple.append(item)

        current_question_dict["ca_attr"] = [item[1] for item in current_question_ca_triple]
        current_question_dict["ca_answ"] = [item[-1] for item in current_question_ca_triple]

        question_dict[data_dict["questions"][i]] = current_question_dict

    return question_dict


        # # Get the candidate answers
        # for item in candidate_triple:
        #     ques_attrs_answs.append(
        #         (entity, item[1], item[-1])
        #     )
            # candidate_attributes.append(item[1])
            # candidate_answers.append(item[-1])

    # return


def pred_final_answers(question_dict: dict, qa_model: str):

    # For each question, predict its answer status.
    pred_answers = []
    questions = list(question_dict.keys())
    for item in questions:
        print(item)
        current_question_answ_stat = predict_candidate_answer(item, question_dict[item]["ca_answ"], qa_model)
        current_question_final_answ = get_final_answer(item, question_dict[item]["ca_answ"], current_question_answ_stat, question_dict[item]["ca_attr"])
        print(current_question_final_answ)
        pred_answers.append(current_question_final_answ)

    return pred_answers

    # question_answer_list = []
    # for current_question in list(question_dict.keys()):

        # for item in question_dict[current_question]["ca_answ"]:

#
#     # Load the Tokenizer and model.
#     tokenizer = BertTokenizer.from_pretrained(qa_model)
#
#     questions_encoded = tokenizer(data_dict["questions"], truncation=True, padding=True, max_length=512)
#
#     input_dataset = Dataset(questions_encoded, [[0] for i in range(len(data_dict["questions"]))])


if __name__ == '__main__':
    data_dict = extract_three_type_data("../data/raw_data/dev.txt")

    question_dict = get_candidate_attr_answer(data_dict)

    pred_answers = pred_final_answers(question_dict, "../bert_qa_model")

    with open("dev_pred_answers.log", "w", encoding="utf-8") as f_obj:
        for item in pred_answers:
            f_obj.write(item)
            f_obj.write("\n")

    true_answers = data_dict["answers"]

    report = classification_report(true_answers, pred_answers, digits=5)

    print(report)

    with open("dev_result.log", "w", encoding="utf-8") as file:
        file.write(report)

