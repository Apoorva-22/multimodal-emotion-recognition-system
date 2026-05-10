import torch
import torch.nn as nn
from transformers import DistilBertModel


class TextEmotionModel(nn.Module):
    def __init__(self):
        super(TextEmotionModel, self).__init__()

        self.bert = DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        )

        self.fc1 = nn.Linear(768, 768)   # embedding
        self.fc2 = nn.Linear(768, 5)     # classification

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_embedding = outputs.last_hidden_state[:, 0, :]

        embedding = self.fc1(cls_embedding)

        output = self.fc2(embedding)

        return output, embedding