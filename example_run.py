import torch
from torch import nn
import json
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, recall_score, precision_score


from hbert_classifier import HurtBertEncodingsClassifier


with open('data/olid.json', 'r') as f:
    train_set = json.load(f)

with open('data/olid_test_levela.json', 'r') as f:
    test_set = json.load(f)

train_x = train_set['sentences']
train_y = train_set['labels_a']

test_x = test_set['sentences']
test_y = test_set['labels']


accuracy = []
f1 = []
precision = []
recall = []

lexicon =''   #lexicon file eg. hurtlex_EN.tsv

for i in range(5):

    model = HurtBertEncodingsClassifier('embedding', lexicon=lexicon) # 'encoding', 'embedding', 'base'


    model.fit(train_x, train_y, epochs=5)
    preds = model.predict(test_x).view(-1).cpu().tolist()

    accuracy.append(accuracy_score(test_y, preds))
    f1.append(f1_score(test_y, preds))
    precision.append(recall_score(test_y, preds))
    recall.append(precision_score(test_y, preds))


with open('experiments_results/HurBERT_embedding.json', 'w') as f:
    json.dump({'accuracy': accuracy,
               'f1': f1,
               'precision': precision,
               'recall': recall}, f)
