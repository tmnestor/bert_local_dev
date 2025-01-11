import torch
from torch import nn

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # print(f"{outputs=}")
        pooled_output = outputs.pooler_output
        # print(f"{pooled_output=}")
        x = self.dropout(pooled_output)
        # print(f"{x=}")
        logits = self.fc(x)
        # print(f"{logits=}")
        return logits