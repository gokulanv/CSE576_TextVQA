# TextVQA 2020 Fb Challenge

Recent works on TextVQA focus on reading and understanding text in the image and answering questions. In this work we analyze the state-of-art models for TextVQA, finding their bottlenecks, points of failure and propose an improved model based on our findings. We propose a caption based modality approach to the visual question task. The embedding of the generated Captions and the question is then fed as inputs to the question answering model trained on SQUAD to generate the final answer. As part of this project, we also  test our approach against existing models on the TextVQA dataset and analyze how it performs.

The project contains the Pythia demos for Image Captioning and Text Visual Question Answering. 

pythia_demo.ipynb  - Demo notebook for TextVQA 

pythia_textcaps_demo.ipynb  - Demo notebook for Image Captioning

BERT-Squad.ipynb - BERT SQuAD fine tuned model inference
