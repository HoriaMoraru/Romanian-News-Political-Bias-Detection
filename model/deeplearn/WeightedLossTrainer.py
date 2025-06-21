from transformers import Trainer
import torch

class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        weights = inputs.pop("weight")

        outputs = model(**inputs)
        logits = outputs.logits

        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        losses = loss_fn(logits, labels)
        weighted_loss = torch.mean(losses * weights)

        return (weighted_loss, outputs) if return_outputs else weighted_loss
