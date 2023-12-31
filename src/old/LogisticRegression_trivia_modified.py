
#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
with open('../data/processed/triviaqa_train.json', 'r', encoding='utf-8') as file:
    data = json.load(file)['data']

# smaller sample
sample_size = 10000
sampled_data = np.random.choice(data, size=sample_size, replace=False)


#%%

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

#%%
'''
Plan 
1. tokenize the sampled questions using distilbert-base-uncased tokenizer
2. using TFiDF Vectorizer, transform them to TFiDF vector
'''
# part 1. 

#%%



#%%
tfidf = TfidfVectorizer(tokenizer=tokenizer, stop_words='', lowercase=False)    

# Extract features and labels
sequences = []
next_words = []

# Collect text for fitting the TF-IDF vectorizer
all_text = []
BagofWords = set()
for item in sampled_data:
    for paragraph in item['paragraphs']:
        context = paragraph['context']
        tokenizer.tokenize(context)
        for qa in paragraph['qas']:
            if 'answers' in qa and qa['answers']:
                answer_text = qa['answers'][0]['text'].lower()
                words = answer_text.split()
                for i in range(len(words) - 1):
                    sequences.append(" ".join(words[:i+1]))
                    next_words.append(words[i+1])

# Fit the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(sequences)

# Transform the text into TF-IDF features
features = tfidf_vectorizer.transform(sequences).toarray()

# Encode labels using LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(next_words)

# Convert lists to numpy arrays
X = np.array(features)
y = label_encoder.transform(next_words)
#%%



#%%

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert numpy arrays to PyTorch tensors
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.long)
X_test_torch = torch.tensor(X_test, dtype=torch.float32)
y_test_torch = torch.tensor(y_test, dtype=torch.long)

# Define Logistic Regression model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        output = F.log_softmax(x, dim=1)
        return output

# Initialize model, loss function, optimizer
input_dim = X_train_torch.shape[1]
output_dim = len(label_encoder.classes_)
model = LogisticRegressionModel(input_dim, output_dim)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=0.1)

# Training the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_torch)
    loss = criterion(outputs, y_train_torch)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Evaluation on test data
model.eval()
with torch.no_grad():
    outputs = model(X_test_torch)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test_torch).sum().item() / len(y_test_torch)
    print(f'Accuracy on test set: {accuracy * 100:.4f}%')
