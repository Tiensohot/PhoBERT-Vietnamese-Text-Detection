# model.py
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

class TransformerCLS4Concat(nn.Module):
    def __init__(self, model_name: str = "vinai/phobert-base", num_labels: int = 2, dropout: float = 0.2):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        hidden = self.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels)
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            return_dict=True
        )
        hs = out.hidden_states
        cls_concat = torch.cat([hs[-i][:, 0, :] for i in range(1, 5)], dim=-1)
        logits = self.classifier(cls_concat)
        return {"logits": logits}