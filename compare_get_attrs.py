from extract_attribute.eval_bert_for_attribute import relation_from_extract_attribute
from application.kbqa_application import get_final_attr, get_candidate_attr_answer
from application.eval_raw_data import extract_three_type_data

if __name__ == '__main__':
    data_dict = extract_three_type_data("../data/raw_data/test.txt")

    question_dict = get_candidate_attr_answer(data_dict)

    print(question_dict)
