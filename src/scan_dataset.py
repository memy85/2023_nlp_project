'''
This code we scan the datasets that we used to do the modeling

'''

#%% This part is for the children book dataset

from pathlib import Path
from datasets import load_dataset

dataset = load_dataset('json', data_files="../data/processed/trivia.json", split='train[:10000]')
trivia_dataset = dataset.train_test_split(test_size=0.2)

dataset = load_dataset("cbt", 'CN', split='train[:10000]')
cbt_dataset = dataset.train_test_split(test_size=0.2)


#%%

''' 
first we analyze the triviaqa dataset
'''
print("the train size : ", len(trivia_dataset['train']),
      "\nthe test sie : ", len(trivia_dataset['test']))


#%%
context = trivia_dataset['train'][1]['context']
answer = trivia_dataset['train'][1]['qas'][0]['answers']
question = trivia_dataset['train'][1]['qas'][0]['question']

#%%

#%%
trivia_dataset['train'][0]

#%%
print(f"the context : \n {context}")
print("\n")
print(f"the question : \n {question}")
print("\n")
print(f"the context : {answer}")

#%%


'''
Now we analyze the children book dataset
'''
#%%
print("the train size : ", len(cbt_dataset['train']),
      "\nthe test sie : ", len(cbt_dataset['test']))

#%%
context = " ".join(cbt_dataset['train'][0]['sentences'])
answer = cbt_dataset['train'][0]['answer']
question = cbt_dataset['train'][0]['question']
options = cbt_dataset['train'][0]['options']

#%%
print(f"the context : \n {context}")
print("\n")
print(f"the question : \n {question}")
print("\n")
print(f"the answer : {answer}")
print(f"\n")
print(f"the options : {options}")
#%%





# %%
