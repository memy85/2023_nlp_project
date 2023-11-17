
#%%
from transformers import pipeline, QuestionAnsweringPipeline
from transformers import GPT2Tokenizer, GPT2ForQuestionAnswering
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments
import pandas as pd
from metrics import compute_metrics
from preprocess import preprocess_trivia

tokenizer = GPT2Tokenizer.from_pretrained("gpt2", return_tesnors='pt')
model = GPT2ForQuestionAnswering.from_pretrained("gpt2")

#%%
from datasets import load_dataset

dataset = load_dataset("cbt", 'raw')
#%%
dataset['train']['content'][0]

#%% After the training, you can output

# preds_outputs = trainer.predict(emotions_encoded["validation"])
# preds_outputs.metrics

#%%

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
#%%
tokenizer.pad_token = tokenizer.eos_token

#%%
# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True,
            max_length = 33)

tokenized_dataset = dataset.map(tokenize_function, batched=True)


#%%
# Define training arguments
training_args = TrainingArguments(
    output_dir="./gpt2_triviaqa",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=1000,
    save_total_limit=2,
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"]
)

#%%
# Start training
trainer.train()

