import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification

class UDA(nn.Module):
    def __init__(self, num_labels, model_name='bert-base-uncased', lambda_u=1.0, T=0.9):
        super(UDA, self).__init__()
        self.num_labels = num_labels
        self.lambda_u = lambda_u
        self.T = T
        
        # Load pretrained model
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.linear = nn.Sequential(nn.Linear(768, 128),
                                    nn.Tanh(),
                                    nn.Linear(128, num_labels))
        

    def forward(self, x, labeled=True, targets=None, epoch_progress=0.0, tsa_schedule='linear'):
        # x is expected to be (input_ids, attention_mask) or just input_ids
        if isinstance(x, tuple):
            input_ids, attention_mask = x
        else:
            input_ids = x
            attention_mask = torch.ones_like(input_ids)
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=targets
        )
        
        # if labeled and targets is not None:
        #     # Apply TSA if this is supervised forward pass
        #     loss = self.apply_tsa(outputs.loss, outputs.logits, targets, epoch_progress, tsa_schedule)
        #     return loss, outputs.logits
        return outputs.logits

    # def apply_tsa(self, loss, logits, labels, progress, schedule='linear'):
    #     if schedule == 'none':
    #         return loss
        
    #     probs = F.softmax(logits, dim=-1)
    #     correct_probs = torch.gather(probs, 1, labels.unsqueeze(1)).squeeze()
        
    #     if schedule == 'linear':
    #         threshold = progress
    #     elif schedule == 'exp':
    #         threshold = torch.exp((progress - 1) * 5)
    #     elif schedule == 'log':
    #         threshold = 1 - torch.exp((-progress) * 5)
    #     else:
    #         raise ValueError(f'Unknown TSA schedule: {schedule}')
        
    #     mask = (correct_probs < threshold).float()
    #     masked_loss = loss * mask
        
    #     if mask.sum() > 0:
    #         return masked_loss.sum() / mask.sum()
    #     return masked_loss.mean()