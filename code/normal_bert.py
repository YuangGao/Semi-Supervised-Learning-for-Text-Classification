import torch
import torch.nn as nn
from transformers import BertModel


class ClassificationBert(nn.Module):
    def __init__(self, num_labels=2):
        super(ClassificationBert, self).__init__()
        # Load pre-trained bert model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Sequential(nn.Linear(768, 128),
                                    nn.Tanh(),
                                    nn.Linear(128, num_labels))

    def forward(self, x, length=256):
        # Encode input text
        outputs = self.bert(x)
        # Get the last hidden state
        last_hidden_state = outputs.last_hidden_state
        # Take mean of the last hidden state along the sequence dimension
        pooled_output = torch.mean(last_hidden_state, dim=1)
        # Use linear layer to do the predictions
        predict = self.linear(pooled_output)

        return predict
