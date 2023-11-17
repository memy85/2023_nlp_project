
from torch import nn
from transformers import Trainer


class TriviaTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs = False) :
        labels = inputs.pop("answers")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_ft = nn.CrossEntropyLoss()
        loss = loss_ft(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
