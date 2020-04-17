import json
import xlwt
from xlwt import Workbook
from statistics import mode

ff = open('TextVQA_Rosetta_OCR_v0.2_train.json',)
ff = open('TextVQA_Rosetta_OCR_v0.2_val.json',)
data0 = json.load(ff)
ocr_map = {}

j = 1
for i in data0['data']:
    answers = ""
    for c, k in enumerate(i['ocr_tokens']):
        if c != 0:
            answers += ", "
        answers += k
    ocr_map[i['image_id']] = answers
    j += 1


wb = Workbook()
sheet1 = wb.add_sheet('Sheet 1')
f = open('TextVQA_0.5.1_train.json',)
# f = open('TextVQA_0.5.1_val.json',)
data = json.load(f)
sheet1.write(0, 0, 'Question')
sheet1.write(0, 1, 'url')
sheet1.write(0, 2, 'flickr_300k_url')
sheet1.write(0, 3, 'image_id')
sheet1.write(0, 4, 'Answer')
sheet1.write(0, 5, 'OCR')

j = 1
for i in data['data']:
    sheet1.write(j, 0, i['question'])
    sheet1.write(j, 1, i['flickr_original_url'])
    sheet1.write(j, 2, i['flickr_300k_url'])
    sheet1.write(j, 3, i['image_id'])
    answers = []
    for k in i['answers']:
        answers.append(k)
    sheet1.write(j, 4, max(answers, key=answers.count))
    sheet1.write(j, 5, ocr_map[i['image_id']])
    j += 1

wb.save('ErrorAnalysis3.xls')
f.close()
