import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW
import numpy as np
from sklearn.metrics import accuracy_score

# if submission == False => splits training data into train and val
# for the final submission => set it true to train the model with the complete training data
submission = True

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# get train and test data
train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")

# get the structure of the submission file
test_dfLabels = pd.read_csv("./data/sample_submission.csv")
pd.set_option('display.max_columns',None)

# preprocess input data
train_texts = train_df['text'].tolist()
test_texts = test_df['text'].tolist()

# preprocess ground truth data
train_labels = train_df['target'].tolist()

# split the training data into train and val for improving hyperparameters
if not submission:
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

# init tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
if not submission:
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

class TweetDatasetTest(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

class TweetDatasetTrain(TweetDatasetTest):
    def __init__(self, encodings, labels):
        super(TweetDatasetTrain, self).__init__(encodings)
        self.labels = labels

    # overwrite function
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# preprocess the dataset
train_dataset = TweetDatasetTrain(train_encodings, train_labels)
if not submission:
    val_dataset = TweetDatasetTrain(val_encodings, val_labels)
test_dataset = TweetDatasetTest(test_encodings)

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.to(device)

loaderTrain = DataLoader(train_dataset, batch_size=64, shuffle=True)
if not submission:
    loaderVal = DataLoader(val_dataset, batch_size=64, shuffle=True)

# init optimizer
optim = AdamW(model.parameters(), lr=5e-5)

# train the model
for epoch in range(2):
    for batch in loaderTrain:
        model.train()
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()

    # if submission is false the training data is splitted into train and val
    if not submission:
        model.eval()
        acc = []
        for batch in loaderVal:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            labels = labels.detach().cpu().numpy()
            outputs = torch.argmax(outputs[1], dim=1).cpu().detach().numpy()
            acc.append(accuracy_score(labels, outputs))
        print("acc eval: ", np.mean(acc))

# run the model on the test data
out=[]
model.eval()
for i in test_dataset:
    input_ids = torch.unsqueeze(i['input_ids'], 0).to(device)
    attention_mask = torch.unsqueeze(i['attention_mask'], 0).to(device)
    outputs = model(input_ids, attention_mask=attention_mask)
    outputs = torch.argmax(outputs[0], dim=1).cpu().detach().numpy()
    [out.append(x) for x in outputs]

# save trained model
torch.save(model.state_dict(), "models/disasterTweets.pth")

# write submission file
test_dfLabels['target'] = out
test_dfLabels.to_csv('results/submission.csv', index=False)
