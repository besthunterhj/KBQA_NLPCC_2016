import numpy as np
import torch
from transformers import BertTokenizer, BertForTokenClassification, Trainer
from test_datasets import Dataset


def predict_entity(sentence: str, ner_model_path: str) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model, tokenizer and trainer.
    tokenizer = BertTokenizer.from_pretrained(ner_model_path)
    model = BertForTokenClassification.from_pretrained(ner_model_path).to(device)
    trainer = Trainer(model)

    # Encode the sentence.
    sentence_encoded = tokenizer([sentence], truncation=True, padding=True, max_length=512)

    # Init the prediction list of NER index
    prediction_list = [[0 for i in range(len(sentence_encoded["input_ids"][0]))]]

    # Build the Dataset
    input_dataset = Dataset(sentence_encoded, prediction_list)
    sentence = sentence.replace(" ", "")

    # Predict from Trainer
    pred_output_matrix = trainer.predict(input_dataset).predictions
    pred_output = np.argmax(pred_output_matrix, axis=2).tolist()

    # Get the Entity from indexes
    beg_index = 0
    end_index = 0
    for i in range(1, len(pred_output[0])):
        if pred_output[0][i] == 1:
            beg_index = i - 1  # [CLS] is the first index of the list, so need to - 1

        if pred_output[0][i] == 2 and pred_output[0][i + 1] == 0:
            end_index = i
            break

    return sentence[beg_index:end_index]


def predict_entities(sentence_list: list, ner_model_path: str) -> list:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model, tokenizer and trainer.
    tokenizer = BertTokenizer.from_pretrained(ner_model_path)
    model = BertForTokenClassification.from_pretrained(ner_model_path).to(device)
    trainer = Trainer(model)

    # Encode the sentence.
    sentence_encoded = tokenizer(sentence_list, truncation=True, padding=True, max_length=512)

    # print(sentence_encoded)
    # Init the prediction list of NER index
    prediction_list = [[0 for j in range(len(sentence_encoded["input_ids"][i]))] for i in range(len(sentence_list))]

    # print(prediction_list)
    # Build the Dataset
    input_dataset = Dataset(sentence_encoded, prediction_list)

    # Predict from Trainer
    pred_output_matrix = trainer.predict(input_dataset).predictions
    pred_output = np.argmax(pred_output_matrix, axis=2).tolist()


    # Get the Entity from indexes
    pred_results = []
    for index in range(len(pred_output)):
        beg_index = 0
        end_index = 0

        for i in range(1, len(pred_output[index])):
            if pred_output[index][i] == 1:
                beg_index = i - 1  # [CLS] is the first index of the list, so need to - 1

            if pred_output[index][i] == 2 and pred_output[index][i + 1] == 0:
                end_index = i
                break

        pred_results.append(sentence_list[index][beg_index:end_index])

    return pred_results

if __name__ == "__main__":
    test_list = [
        "圣殿骑士团的规模有多大啊？",
        "广东外语外贸大学的校区有多少个？",
        "广东的省会是哪个城市？"
                 ]

    results = predict_entities(test_list, "../bert_ner_model")

    print(results)

