import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch

# data loading and inspection

data = pd.read_csv('2020-12-31-DynamicallyGeneratedHateDataset-entries-v0.1.csv')

data.drop(columns=['id', 'model_wrong', 'db.model_preds', 'annotator', 'round', 'status', 'split', 'Unnamed: 0', 'type'], inplace=True)

label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

print(data['label'].value_counts())

# perform the train-test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42
)

print("Training samples:", len(X_train))
print("Test Samples:", len(X_test))

# Since the data is slightly imbalanced, I am going to use class weights to address this issue

# Calculating the class weights

train_class_counts = y_train.value_counts()
total_train_samples = len(y_train)
train_class_weights = [total_train_samples / train_class_counts[i] for i in range(len(train_class_counts))]
train_class_weights = torch.tensor(train_class_weights, dtype=torch.float)
print("Class Weights:", train_class_weights)

# Tokenize the data

# Tokenizing the data is a critical step in natural language processing (NLP) tasks 
# when using transformer-based models like DistilBERT. 

# Transformers like DistilBERT cannot process raw text directly. 
# They require numerical input, such as integers or tensors. 
# Tokenization converts text into numerical representations that the model can process.

# Example:
# For the text: "I love NLP", tokenization might produce:

# Tokens: ["I", "love", "NLP"]
# Token IDs: [100, 2028, 2562]

from transformers import DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_data(texts, labels, tokenizer, max_length=128):
    tokens = tokenizer(
        texts.tolist(),
        max_length = max_length,
        padding = 'max_length',
        truncation = True,
        return_tensors = "pt"
    )
    return tokens, torch.tensor(labels.tolist())

train_tokens, train_labels = tokenize_data(X_train, y_train, tokenizer)
test_tokens, test_labels = tokenize_data(X_test, y_test, tokenizer)

print(train_tokens)
print(train_labels)
print(test_labels)
print(test_tokens)

# saving these tokens for further use

torch.save(train_tokens, "tokenization/train_tokens.pt")
torch.save(train_labels, "tokenization/train_labels.pt")
torch.save(test_tokens, "tokenization/test_tokens.pt")
torch.save(test_labels, "tokenization/test_labels.pt")
 
# # Using Hugging Face's Trainer for easy training
# from torch.utils.data import Dataset

# class HateSpeechDataset(Dataset):
#     def __init__(self, tokens, labels):
#         self.tokens = tokens
#         self.labels = labels

#     def __len__(self):
#         return len(self.labels)
    
#     def __getitem__(self, index):
#         return {
#             "input_ids": self.tokens['input_ids'][index],
#             "attention_mask": self.tokens['attention_mask'][index],
#             "labels": self.labels[index]
#         }
    

# train_dataset = HateSpeechDataset(train_tokens, train_labels)
# test_dataset = HateSpeechDataset(test_tokens, test_labels)

# training_args = TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=3,
#     per_device_train_batch_size=16,
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     logging_dir='./logs',
#     logging_steps=10,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset
# )

# # model training, no need to run everytime, run when you download the code for the first time.
# # trainer.train()

# eval_results = trainer.evaluate()
# print(eval_results)
            
# tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# model = DistilBertForSequenceClassification.from_pretrained("./results/checkpoint-6096")

# text = ["you are an asshole", "This is a neutral comment", "dalits are lowlife scum"]
# tokens = tokenizer(
#     text, 
#     max_length=128, 
#     padding='max_length', 
#     truncation=True, 
#     return_tensors="pt"
# )

# outputs = model(**tokens)
# predictions = torch.argmax(outputs.logits, dim=1)
# print(predictions)


