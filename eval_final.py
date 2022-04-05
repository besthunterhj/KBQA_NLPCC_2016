import re

from sklearn.metrics import classification_report


final_answers = []
with open("pred_answers.log", "r", encoding="utf-8") as file:
    lines = file.readlines()

    for line in lines:
        final_answers.append(line.strip())

print(len(final_answers))

true_answers = []
pattern = re.compile("id=[0-9]{1,8}> ")
with open("../data/raw_data/train.txt", "r", encoding="utf-8") as f_obj:
    lines = f_obj.readlines()

    for line in lines:
        if "<answer" in line:
            answer = pattern.split(line)[-1].strip()
            true_answers.append(answer)

report = classification_report(true_answers, final_answers, digits=5)
print(report)