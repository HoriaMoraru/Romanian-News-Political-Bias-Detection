from transformers import Trainer
import torch.nn.functional as F

class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        weights = inputs.pop("weight")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = F.cross_entropy(logits, labels, reduction='none')
        weighted_loss = (loss * weights).mean()
        return (weighted_loss, outputs) if return_outputs else weighted_loss
