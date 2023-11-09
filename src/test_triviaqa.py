
#%%
from transformers import pipeline
from transformers import OpenAIGPTTokenizer, OpenAIGPTModel
import pandas as pd
from metrics import compute_metrics

tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
model = OpenAIGPTModel.from_pretrained("openai-gpt")

#%%
from datasets import load_dataset

dataset = load_dataset("trivia_qa", "unfiltered")

#%%

def tokenize(batch) :
    return tokenizer(batch['text'], padding=True, truncation=True)

dataset_encoded = dataset.map(tokenize, batched=True, batch_size=None)
#%%

from transformers import Trainer, TrainingArguments

batch_size = 64
logging_steps = len(dataset_encoded["train"])
model_name = f"openai-gpt-finetuned-triviaqa"
training_args = TrainingArguments(output_dir= model_name,
                                  num_train_epochs = 2,
                                  learning_rate=2e-5,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size,
                                  weight_decay=0.01,
                                  evaluation_strategy="epoch",
                                  disable_tqdm=False,
                                  logging_steps = logging_steps,
                                  push_to_hub=False, # if you want to push to the hub
                                  log_level="error")


#%% We now use the Trainer
from transformers import Trainer

trainer = Trainer(model=model, args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=dataset_encoded["train"],
                  eval_dataset=dataset_encoded["validation"],
                  tokenizer=tokenizer)

trainer.train()

#%% After the training, you can output

preds_outputs = trainer.predict(emotions_encoded["validation"])
preds_outputs.metrics

