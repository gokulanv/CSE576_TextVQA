import json
import argparse
import torch
import urllib
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
QUESTIONS_FILE_URL = "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json"


def bert_squad(image_caption, question):
    inputs = tokenizer.encode_plus(question, image_caption, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer_start_scores, answer_end_scores = model(**inputs)
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer


# load squad model
tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model = AutoModelForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--image_id", help="refer to image_id from val dataset of open_images, type all to process all images")
parser.add_argument("--captions_file", help="captions json output from model", nargs='?', default='pythia/data/m4c_textcaps_run_val_2020-04-19T02_19_58.json')
parser.add_argument("--n_questions", help='mention # of questions you wish to ask <int>, type all to use the questions json as input')
args = parser.parse_args()

# load captions data
captions_file = open(args.captions_file)
captions_data = json.load(captions_file)
images_available = [record["image_id"] for record in captions_data]

# load questions data
if args.n_questions == 'all':
    questions = []
    with urllib.request.urlopen(QUESTIONS_FILE_URL) as url:
        questions_data = json.loads(url.read().decode())

    for i in questions_data['data']:
        temp = {'question': i['question'], 'flickr_original_url': i['flickr_original_url'],
                'flickr_300k_url': i['flickr_300k_url'], 'image_id': i['image_id']}
        questions.append(temp)


# main
if args.image_id in images_available:
    if args.n_questions == 'all':
        # TODO implement questions_all
        pass
    elif args.n_questions.isdigit():
        # TODO show the image in dialog box

        # fetch image caption
        for record in captions_data:
            if args.image_id == record["image_id"]:
                image_caption = record["caption"]
        print("Generated Caption: ", image_caption)

        for _ in range(int(args.n_questions)):
            question = input("Question: ")
            answer = bert_squad(image_caption, question)
            print("Answer: ", answer)
elif args.image_id == 'all':
    # TODO implement images all
    # ignore n_questions and just all images for all their corresponding questions
    pass
else:
    print("Image id not found, try again")



#TODO run.py --image_id <--all> --M4CTextcaps_output.json -- textvqa_questions.json
