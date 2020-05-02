import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model = AutoModelForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

captions_file = open('pythia/data/m4c_textcaps_run_val_2020-04-19T02_19_58.json')

captions_data = json.load(captions_file)

images_available = [record["image_id"] for record in captions_data]

parser = argparse.ArgumentParser()


parser.add_argument("--image_id", help="refer to image_id from val dataset of open_images")
args = parser.parse_args()


def bert_squad(image_caption, question):
    inputs = tokenizer.encode_plus(question, image_caption, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer_start_scores, answer_end_scores = model(**inputs)
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer

print(images_available)

if args.image_id in images_available:
    question = input("Question: ")
    for record in captions_data:
        if args.image_id == record["image_id"]:
            image_caption = record["caption"]

    answer = bert_squad(image_caption, question)
    print("Answer: ", answer)
else:
    print("Image id not found, try again")
