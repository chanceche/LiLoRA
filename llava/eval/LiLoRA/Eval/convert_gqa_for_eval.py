import argparse
import json


parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str)
parser.add_argument("--dst", type=str)
args = parser.parse_args()

all_answers = []
for line in open(args.src):
    res = json.loads(line)
    all_answers.append({
        "questionId": res["question_id"],
        "prediction": res["text"].rstrip(".").lower(),
    })

with open(args.dst, "w") as f:
    json.dump(all_answers, f)
