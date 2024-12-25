import torch
import torch.nn as nn

class MultiTaskModel(nn.Module):
    def __init__(self, input_model):
        super(MultiTaskModel, self).__init__()
        self.bert = input_model
        self.span_classifier = nn.Linear(1024, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        last_hidden_state = output[0]
        last_hidden_state = self.dropout(last_hidden_state)
        span_logits = self.span_classifier(last_hidden_state)

        span_logits = span_logits.permute(0, 2, 1)
        span_logits = torch.sigmoid(span_logits)
        span_logits = span_logits.permute(0, 2, 1)

        return  span_logits