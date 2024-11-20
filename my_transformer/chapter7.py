import torch
from torch import nn
from transformers import AutoModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BertforCls(nn.Module):
    def __init__(self):
        super(BertforCls, self).__init__()
        self.bert_encoder = AutoModel.from_pretrained(checkpoint)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 2)

    def forward(self, x):
        bert_out = self.bert_encoder(x)
        cls_vectors = bert_out.lase_hidden_state[:, 0, :]
        cls_vectors = self.dropout(cls_vectors)
        logits = self.classifier(cls_vectors)
        return logits
    
model = BertforCls().to(device)
print(model)



