import torch.cuda
from transformers import BertForSequenceClassification, BertTokenizer, AutoTokenizer, DataCollatorWithPadding, \
    TrainingArguments, Trainer, IntervalStrategy
from test_datasets import Dataset

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logging_dir = "./output_log"


def split_pair_data(data_set: str):
    pairs = []
    labels = []

    with open(data_set, "r", encoding="utf-8") as file:
        lines = file.readlines()

        for line in lines:
            if line == "\n":
                continue

            current_question, current_answer = line.split("\t")[0].strip(), line.split("\t")[1].strip()
            current_label = line.split("\t")[-1].strip()

            current_pair = (current_question, current_answer)
            labels.append(int(current_label))
            pairs.append(current_pair)

    return pairs, labels


def main(trainset_path, devset_path, model_path):
    # Load the data.
    train_set_pairs, train_set_labels = split_pair_data(trainset_path)
    dev_set_pairs, dev_set_labels = split_pair_data(devset_path)

    # Load the Tokenizer and Model.
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2).to(device)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Construct the format.
    train_text, dev_text = [], []
    for pair in train_set_pairs:
        item = [pair[0], pair[1]]
        train_text.append(item)

    for pair in dev_set_pairs:
        text = [pair[0], pair[1]]
        dev_text.append(text)

    # Encode the text.
    train_encoded_token = tokenizer(train_text, max_length=512, truncation=True, padding=True)
    dev_encoded_token = tokenizer(dev_text, max_length=512, truncation=True, padding=True)

    # Build the datasets.
    train_dataset = Dataset(train_encoded_token, train_set_labels)
    dev_dataset = Dataset(dev_encoded_token, dev_set_labels)

    # Init the Args and Trainer.
    args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_steps=0,  # number of warmup steps for learning rate scheduler
        gradient_accumulation_steps=1,
        adam_epsilon=1e-8,
        logging_dir=logging_dir,  # directory for storing logs
        logging_steps=10,
        # Eval steps. Will save a ckpt every `eval_steps`.
        eval_steps=1000,
        save_steps=1000,
        metric_for_best_model="loss",
        evaluation_strategy="steps",
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,)

    trainer.train()

if __name__ == "__main__":
    main("train.txt", "dev.txt", "../bert-base-chinese")

