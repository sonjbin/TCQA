# Time-Aware Representation Learning for Time-Sensitive Question Answering



## Abstract
Time is one of the crucial factors in real-world question answering (QA) problems. However, language models have difficulty understanding the relationships between time specifiers, such as 'after' and 'before', and numbers, since existing QA datasets do not include sufficient time expressions. To address this issue, we propose a Time-Context aware Question Answering (TCQA) framework. We suggest a Time-Context dependent Span Extraction (TCSE) task, and build a time-context dependent data generation framework for model training. Moreover, we present a metric to evaluate the time awareness of the QA model using TCSE. The TCSE task consists of a question and four sentence candidates classified as correct or incorrect based on time and context. The model is trained to extract the answer span from the sentence that is both correct in time and context. The model trained with TCQA outperforms baseline models up to 8.5 of the F1-score in the TimeQA dataset. 

## TCSE task
<p align="center">
<img src="./images/TCSE.png" width="70%" height="42.35%" title="TCSE"/>
</p>

##Repo Structure
- 

## Requirements
- Python 3.8.2
- PyTorch 1.10.2+cu113
- transformers 4.10.2

## Getting Started
### Data Generation
You need to complete the three parts in order.
1. Generate context template 
```bash
python generate_TCSE.py --loadtc False --loadq False --data train
```
2. Generate question template using context
```bash
python generate_TCSE.py --loadtc True --loadq False --data train
```
3. Generate TCSE dataset using question-context template
```bash
python generate_TCSE.py --loadtc True --loadq True --data train
```

### Train on TimeQA with TCQA
BigBird
```bash
python -m BigBird.main model_id=nq dataset=hard cuda=0 mode=train TCSE=True k=1.0 CRL=True k_crl=0.5
```
BERT, RoBERTa, ALBERT
```bash
python -m BigBird.main model_id=[bertbase or robertabase or albertbase] dataset=hard mode=eval use_bert=True max_sequence_length=512 doc_stride=256 TCSE=True k=1.0 CRL=True k_crl=1.0
```
### Test
BigBird
```bash
python -m BigBird.main model_id=nq dataset=hard cuda=0 mode=eval model_path=[YOUR_MODEL]
```
BERT, RoBERTa, ALBERT
```bash
python -m BigBird.main model_id=[bertbase or robertabase or albertbase] dataset=hard mode=eval use_bert=True max_sequence_length=512 doc_stride=256 model_path=[YOUR_MODEL]
```
### TC-score
1. Get output file by removing null classifier
```bash
python -m BigBird.main model_id=nq dataset=hard cuda=0 mode=eval model_path=[YOUR_MODEL] --TCAS True
```
2. Calculate TC-score
```bash
python tcscore --predict_path [OUTPUT FILE]
```

## Code Reference

We referred https://github.com/wenhuchen/Time-Sensitive-QA to implement the code for preprocessing TimeQA benchmark dataset.