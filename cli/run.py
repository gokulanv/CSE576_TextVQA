import json
import argparse
import torch
import urllib
import pickle
from tqdm import tqdm
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
parser.add_argument("--n_questions", help='mention # of questions you wish to ask <int>, type 0 to ask predefined question')
args = parser.parse_args()

# load captions data
captions_file = open(args.captions_file)
captions_data = json.load(captions_file)
images_and_captions = dict()

for record in captions_data:
    images_and_captions[record["image_id"]] = record["caption"]

# load questions data
questions = {}
with urllib.request.urlopen(QUESTIONS_FILE_URL) as url:
    questions_data = json.loads(url.read().decode())

for i in questions_data['data']:
    temp = {'question': i['question'], 'flickr_original_url': i['flickr_original_url'],
            'flickr_300k_url': i['flickr_300k_url'], 'question_id': i['question_id']}
    questions[i['image_id']] = temp


# For single image generations
if args.image_id in images_and_captions:

    print("Generated Caption: ", images_and_captions[args.image_id])

    if int(args.n_questions) == 0:
        question_dict = questions[args.image_id]
        answer = bert_squad(images_and_captions[args.image_id], question_dict['question'])
        print("Answer: ", answer)
    else:
        for _ in range(int(args.n_questions)):
            question = input("Question: ")
            answer = bert_squad(images_and_captions[args.image_id], question)
            print("Answer: ", answer)

# For batch processing
elif args.image_id == 'all':
    result = []
    pbar = tqdm(total=len(images_and_captions))

    for image_id, caption in images_and_captions.items():
        question_dict = questions[image_id]
        question = question_dict['question']
        question_id = question_dict['question_id']
        answer = bert_squad(caption, question)
        pbar.update(1)
        result.append({'question_id': question_id, 'answer': answer})
    pbar.close()
    with open("textvqa_w_caps_val_result.json", "wb") as fp:
        pickle.dump(result, fp)
else:
    print("Image id not found, try again")



# DONE TODO run.py --image_id <--all> --M4CTextcaps_output.json -- textvqa_questions.json
# DONE TODO write a show image script
# DONE TODO implement one image, all questions
# DONE TODO implement all images, all questions
