
from sklearn.metrics import accuracy_score, f1_score
from torchmetrics.text import Perplexity

def compute_metrics(pred) :
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average = "weighted")
    acc = accuracy_score(labels, preds)
    perplexity = Perplexity()
    perp = perplexity(preds, labels)
    return {"accuracy" : acc, "f1" : f1, "perplexity" : perp}
