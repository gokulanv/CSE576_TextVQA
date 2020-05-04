## Run CLI

### Arguments to CLI

usage: run.py [-h] [--image_id IMAGE_ID] [--captions_file [CAPTIONS_FILE]]
               [--n_questions N_QUESTIONS]

optional arguments:
  -h, --help            show this help message and exit
  --image_id IMAGE_ID   refer to image_id from val dataset of open_images,
                        type all to process all images
  --captions_file [CAPTIONS_FILE]
                        captions json output from model
  --n_questions N_QUESTIONS
                        mention # of questions you wish to ask <int>, type all
                        to use the questions json as input

### Sample command
python run.py --image_id 17f184cad1ba4d19 --captions_file pythia/data/m4c_textcaps_run_val_2020-04-19T02_19_58.json --n_question 1

### Dependencies
torch

urllib

transformers