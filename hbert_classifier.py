import torch
from torch import nn

from hurt_encoding import HurtEncoding
from hbert_model import HurtBERT, HurtBERTEmb, BaseBERT

from transformers import BertTokenizer
from torch.utils.data import DataLoader

from tqdm.autonotebook import tqdm, trange

from sentence_transformers import SentenceTransformer, util
from torch.nn.utils.rnn import pad_sequence
import transformers

from input_example import InputExample


class HurtBertEncodingsClassifier():

    def __init__(self, model_type, lexicon):
        
        assert model_type in ['encoding', 'embedding', 'base']

        self.model_type = model_type

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        if model_type == 'encoding':
            self.hurt_lex_encoder = HurtEncoding(hurt_lex_dataset_path='lexicon', method='hurt_bert_encoding', num_categories=17)
            self.hurtbert = HurtBERT()
        elif model_type == 'embedding':
            self.hurt_lex_encoder = HurtEncoding(hurt_lex_dataset_path='lexicon', method='hurt_bert_embedding', num_categories=17)
            self.hurtbert = HurtBERTEmb()
        else:
            self.hurt_lex_encoder = HurtEncoding(hurt_lex_dataset_path='lexicon', method='hurt_bert_encoding', num_categories=17)
            #placeholder
            self.hurtbert = BaseBERT()

    
    def predict(self, sentences, batch_size=32):

        inp_dataloader = DataLoader(sentences, batch_size=batch_size, collate_fn=self.smart_batching_collate_text_only, shuffle=False)
        

        pred_scores = []
        self.hurtbert.eval()
        self.hurtbert.cuda()

        with torch.no_grad():
            for features in inp_dataloader:
                model_predictions = self.hurtbert(**features)['logits']
                logits = (model_predictions>1).float()
                pred_scores.extend(logits)

        return torch.stack(pred_scores)

    def smart_batching_collate_text_only(self, batch):
        lexical_encodings = self.hurt_lex_encoder.encode_sentence(batch)
        tokenized = self.tokenizer(batch, return_tensors="pt", padding=True)

        for name in tokenized:
            tokenized[name] = tokenized[name].cuda()

        if self.model_type == 'encoding':
            tokenized['encoding'] = torch.stack(lexical_encodings).cuda()
        elif self.model_type == 'embedding':
            tokenized['encoding'] = pad_sequence(lexical_encodings).permute(1,0,2).cuda()
        else:
            pass

        return tokenized
    


    def fit(self, sentences, labels, epochs=1, batch_size=32, lr = 1e-5):
    
        train_samples = [InputExample(texts=sentence, label=label) for sentence, label in zip(sentences, labels)]

        train_dataloader = DataLoader(train_samples, batch_size=batch_size, collate_fn=self.smart_batching_collate, shuffle=True)

        self.hurtbert.cuda()

        num_train_steps = int(len(train_dataloader) * epochs)

        optimizer = torch.optim.AdamW(self.hurtbert.parameters(), lr=lr)
        loss_fct = nn.BCEWithLogitsLoss()
        activation_fct = nn.Identity()


        for epoch in trange(epochs, desc="Epoch", disable=False):
            training_steps = 0
            self.hurtbert.zero_grad()
            self.hurtbert.train()

            for batch in tqdm(train_dataloader, desc="Iteration", smoothing=0.05, disable=False):
                features = batch[0]
                labels = batch[1]
                model_predictions = self.hurtbert(**features)
                logits = activation_fct(model_predictions.logits)
                logits = logits.view(-1)
                loss_value = loss_fct(logits, labels)
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(self.hurtbert.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()



    def smart_batching_collate(self, batch):
        texts = []
        labels = []
        
        for example in batch:
            texts.append(example.texts)
            labels.append(example.label)


        lexical_encodings = self.hurt_lex_encoder.encode_sentence(texts)
        tokenized = tokenized = self.tokenizer(texts, return_tensors="pt", padding=True)
        labels = torch.tensor(labels, dtype=torch.float).cuda()

        if self.model_type == 'encoding':
            tokenized['encoding'] = torch.stack(lexical_encodings).cuda()
        elif self.model_type == 'embedding':
            tokenized['encoding'] = pad_sequence(lexical_encodings).permute(1,0,2).cuda()
        else:
            pass

        for name in tokenized:
            tokenized[name] = tokenized[name].cuda()

        return tokenized, labels
