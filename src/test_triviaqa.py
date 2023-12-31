
#%%
import datasets
from datasets import load_dataset
from pathlib import Path
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from transformers import DefaultDataCollator
from transformers import Trainer, TrainingArguments
import json
import torch
import evaluate
from tqdm.auto import tqdm
import collections
import numpy as np

project_dir = Path().cwd().parent
data_dir = project_dir.joinpath("data")


max_answer_length = 30
n_best = 20
metric = evaluate.load('squad')

#%%
# with open('../data/processed/triviaqa_train.json','r') as f :
#     data = json.load(f)

# def extractor(paragraph) :
#     return paragraph['paragraphs'][0]

# #%%
# all_data = [extractor(book) for book in data['data']]

# with open('../data/processed/trivia.json', 'w', encoding='utf-8') as f :
#     json.dump(all_data, f)

dataset = load_dataset('json', data_files="../data/processed/trivia.json", split='train[:10000]')
dataset = dataset.train_test_split(test_size=0.2)

#%%

print("---------------------------------------------------------------------------------------")
print("------------------------------- processed dataset -------------------------------------")
print("---------------------------------------------------------------------------------------")

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForQuestionAnswering.from_pretrained('distilbert-base-uncased')

#%%

def preprocess_function(examples):
    questions = [q[0]['question'].strip() for q in examples["qas"]]
    contexts = [context.strip() for context in examples['context']]
    answers = [q[0]['answers'] for q in examples["qas"]]

    inputs = tokenizer(
        questions,
        contexts,
        max_length=512,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs["offset_mapping"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        if len(answers[i]) < 1 :
            start_positions.append(0)
            end_positions.append(0)
            continue
        else :
            answer = answers[i][0]

            start_char = answer["answer_start"]
            end_char = answer["answer_start"] + len(answer["text"])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx

            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


tokenized_dataset = dataset.map(preprocess_function, batched=True, )
tokenized_dataset['train'] = tokenized_dataset['train'].remove_columns(dataset['train'].column_names)
#%%
tokenized_dataset.save_to_disk("../data/processed/tokenized_trivia")

#%%

def compute_metrics(start_logits, end_logits, tokenized,  original):
    '''
    metrics for question answering. 
    The position where the answer starts and ends becomes the target. 
    '''

    predicted_answers = []
    theoretical_answers = []
    for idx, example in enumerate(tqdm(original)):

        context = example["context"]
        original_answer = example['qas'][0]['answers']
        if len(original_answer) < 1:
            continue

        answers = []

        start_logit = start_logits[idx]
        end_logit = end_logits[idx]
        offsets = tokenized["offset_mapping"][idx]

        start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
        end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
        for start_index in start_indexes:
            for end_index in end_indexes:
                # Skip answers that are not fully in the context
                if offsets[start_index] is None or offsets[end_index] is None:
                    continue
                # Skip answers with a length that is either < 0 or > max_answer_length
                if (
                    end_index < start_index
                    or end_index - start_index + 1 > max_answer_length
                ):
                    continue

                answer = {
                    "text": context[offsets[start_index][0] : offsets[end_index][1]],
                    "logit_score": start_logit[start_index] + end_logit[end_index],
                }
                answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": str(idx), "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": str(idx), "prediction_text": ""})
        
        theoretical_answers.append({"id": str(idx), "answers" : original_answer})

    return metric.compute(predictions=predicted_answers, references=theoretical_answers)

#%%
print("------------------------------------------------------------------")
print("--------------------- tokenized the data -------------------------")
print("------------------------------------------------------------------")

data_collator = DefaultDataCollator()

training_args = TrainingArguments(
    output_dir="../model/trivia_distilbert",
    disable_tqdm=False,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=100,
    weight_decay=0.01,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

######################## ##################### ################## #################
######################## ##################### ################## Now we do testing
######################## ##################### ################## #################


#%%
checkpoint = "../model/trivia_distilbert/checkpoint-10000"
model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

#%% load already saved dataset
tokenized_dataset = datasets.load_from_disk('../data/processed/tokenized_trivia')

#%%
torch.cuda.set_device(1)
# torch.cuda.current_device()

#%%
data_collator = DefaultDataCollator()
training_args = TrainingArguments("test-trainer",
                                  per_device_eval_batch_size=8)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    # eval_dataset=tokenized_dataset['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    # compute_metrics=compute_metrics
)

#%%
# ------------------------------------- Testing the results

# sample = tokenized_dataset['test'].select(range(0,100))
sample = tokenized_dataset['test']

#%%
predictions,_, _ = trainer.predict(sample)
start_logits, end_logits = predictions
compute_metrics(start_logits, end_logits, sample, sample)

