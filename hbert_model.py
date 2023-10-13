from transformers import BertModel, BertTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

import torch.nn.functional as F


from torch import nn
import torch

class HurtBERT(nn.Module):
    
    def __init__(self, bert_dense_outpu_size = 64, hurt_lex_encoding_size = 17, bert_model='bert-base-uncased'):
        super(HurtBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_dense = nn.Linear(768, bert_dense_outpu_size)
        self.last_layer  = nn.Linear(bert_dense_outpu_size+hurt_lex_encoding_size, 1)

    
    def forward(self, input_ids, attention_mask, encoding, token_type_ids, labels=None):
        outputs = self.bert(input_ids, attention_mask)
        x = F.relu(self.bert_dense(outputs[1]))
        x = torch.concat((x, encoding), dim=1)
        x = self.last_layer(x)

        loss = None
        if labels is not None:
            loss_fct = nn.BCELoss()
            loss = loss_fct(x.view(-1, 1), labels.view(-1))
        
        return SequenceClassifierOutput(
            loss=loss, 
            logits=x, 
            hidden_states=outputs.hidden_states, 
            attentions=outputs.attentions
            )



class HurtBERTEmb(nn.Module):
    
    def __init__(self, bert_dense_outpu_size = 64, hurt_lex_encoding_size = 17, bert_model='bert-base-uncased'):
        super(HurtBERTEmb, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.num_layers = 1
        self.hidden_size=32
        self.lstm = nn.LSTM(input_size=17, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)

        self.bert_dense = nn.Linear(768, bert_dense_outpu_size)
        self.lstm_dense = nn.Linear(32, 16)
        self.last_layer  = nn.Linear(bert_dense_outpu_size+16, 1)

    
    def forward(self, input_ids, attention_mask, encoding, token_type_ids, labels=None):

        h0 = torch.zeros(self.num_layers, encoding.size(0), self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers, encoding.size(0), self.hidden_size).cuda()

        # Forward propagate LSTM
        out, _ = self.lstm(encoding, (h0, c0))

        out = out[:, -1, :]
        
        outputs = self.bert(input_ids, attention_mask)
        x1 = F.relu(self.lstm_dense(out))

        x = F.relu(self.bert_dense(outputs[1]))
        x = torch.concat((x, x1), dim=1)
        x = self.last_layer(x)

        loss = None
        if labels is not None:
            loss_fct = nn.BCELoss()
            loss = loss_fct(x.view(-1, 1), labels.view(-1))
        
        return SequenceClassifierOutput(
            loss=loss, 
            logits=x, 
            hidden_states=outputs.hidden_states, 
            attentions=outputs.attentions
            )
    

class BaseBERT(nn.Module):
    
    def __init__(self, bert_dense_outpu_size = 64, bert_model='bert-base-uncased'):
        super(BaseBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.bert_dense = nn.Linear(768, 1)

    
    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.bert(input_ids, attention_mask)
        x = self.bert_dense(outputs[1])

        loss = None
        if labels is not None:
            loss_fct = nn.BCELoss()
            loss = loss_fct(x.view(-1, 1), labels.view(-1))
        
        return SequenceClassifierOutput(
            loss=loss, 
            logits=x, 
            hidden_states=outputs.hidden_states, 
            attentions=outputs.attentions
            )