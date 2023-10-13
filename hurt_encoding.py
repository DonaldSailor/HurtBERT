import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class HurtEncoding:
    def __init__(
            self,
            hurt_lex_dataset_path: str,
            method: str,
            num_categories: int,      
            ) -> None:
        
        self._hurt_lex_dataset = pd.read_csv(hurt_lex_dataset_path, sep='\t')
        assert method in ['hurt_bert_encoding', 'hurt_bert_embedding']
        self.method = method
        self.num_categories = num_categories

        if self.method == 'hurt_bert_encoding':
            self._hurt_lex_dataset['label'] = LabelEncoder().fit_transform(self._hurt_lex_dataset['category'])
            assert self._hurt_lex_dataset['label'].max() + 1 == self.num_categories
        
        elif self.method == 'hurt_bert_embedding':
            self._hurt_lex_dataset['label'] = OneHotEncoder(sparse=False).fit_transform(
                self._hurt_lex_dataset['category'].array.reshape(-1, 1)).tolist()
            assert len(self._hurt_lex_dataset['label'][0]) == self.num_categories
    
    def encode(self, tokenized_sentence: torch.Tensor) -> torch.Tensor:
        if self.method == 'hurt_bert_encoding':
            hurt_bert_enc_vector = torch.zeros(self.num_categories)
            for token in tokenized_sentence:
                if token in self._hurt_lex_dataset['lemma'].values:
                    hurt_bert_enc_vector[self._hurt_lex_dataset[self._hurt_lex_dataset['lemma'] == token]['label'].values[0]] += 1
            return hurt_bert_enc_vector

        elif self.method == 'hurt_bert_embedding':
            hurt_bert_emb_vector = torch.zeros(len(tokenized_sentence), self.num_categories)
            for i, token in enumerate(tokenized_sentence):
                if token in self._hurt_lex_dataset['lemma'].values:
                    hurt_bert_emb_vector[i] = torch.tensor(self._hurt_lex_dataset[self._hurt_lex_dataset['lemma'] == token]['label'].values[0])
                else:
                    hurt_bert_emb_vector[i] = torch.zeros(self.num_categories)
            return hurt_bert_emb_vector
        

    def encode_sentence(self, sentences):
        tokenized_sentences = [i.split(' ') for i in sentences]
        lex_sentences = []

        for ts in tokenized_sentences:
            lex_sentences.append(self.encode(tokenized_sentence=ts))

        return lex_sentences