import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer

CTX_LEN = 50
device = torch.device("cuda:3")


rand_seed = 218276150
torch.manual_seed(rand_seed)
torch.cuda.manual_seed(rand_seed)
torch.cuda.manual_seed_all(rand_seed)
np.random.seed(rand_seed)
random.seed(rand_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


bert_model = BertModel.from_pretrained("bert-base-uncased")
for param in bert_model.parameters():
    param.requires_grad = False


# Define SwitchTransformerMoE model
class BertClassifier(nn.Module):
    def __init__(self, base_model, base_model_dim, ctx_len, num_classes=2):
        super(BertClassifier, self).__init__()
        self.base_model = base_model
        self.classifier = nn.Sequential(
            nn.Linear(ctx_len * base_model_dim, 768), nn.Dropout(0.2), nn.ReLU(), nn.Linear(768, num_classes)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_id, mask):
        output = self.base_model(input_ids=input_id, attention_mask=mask).last_hidden_state
        output = output.reshape(output.size(0), -1)
        logits = self.softmax(self.classifier(output))
        return logits


model = BertClassifier(bert_model, 768, ctx_len=CTX_LEN, num_classes=2)


class FARNDataset(Dataset):
    def __init__(self, ctx_len=100, train=True) -> None:
        super().__init__()
        self.ctx_len = ctx_len
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        fake_df = pd.read_csv("Fake.csv")
        true_df = pd.read_csv("True.csv")
        fake_len = len(fake_df)
        true_len = len(true_df)
        fake_split_idx = int(fake_len * 0.7)
        true_split_idx = int(true_len * 0.7)
        if train:
            self.fake_df = fake_df.iloc[:fake_split_idx]
            self.true_df = true_df.iloc[:true_split_idx]
        else:
            self.fake_df = fake_df.iloc[fake_split_idx:]
            self.true_df = true_df.iloc[true_split_idx:]
        self.fake_len = len(self.fake_df)
        self.true_len = len(self.true_df)

    def __len__(self):
        return self.fake_len + self.true_len

    def __getitem__(self, i):
        if i < self.fake_len:
            sent = f"{self.fake_df.iloc[i][0]} {self.fake_df.iloc[i][1]}"
            label = 0
        else:
            i -= self.fake_len
            sent = f"{self.true_df.iloc[i][0]} {self.true_df.iloc[i][1]}"
            label = 1
        encoded = self.tokenizer(sent, padding="max_length", max_length=self.ctx_len, truncation=True, return_tensors="pt")
        # encoded = torch.tensor(encoded, dtype=torch.int64)
        # label = torch.tensor(label, dtype=torch.int64)
        return encoded, np.array(label)


# def farn_collate_fn(batch):
#     input = []
#     target = []
#     for x, y in batch:
#         input.append(x)
#         target.append(y)
#     input = torch.tensor(input, dtype=torch.int64)
#     target = torch.tensor(target, dtype=torch.int64)
#     return input, target


data_loader = DataLoader(FARNDataset(CTX_LEN), batch_size=128, shuffle=True, num_workers=5)


model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)


test_loader = DataLoader(FARNDataset(CTX_LEN, False), batch_size=128, shuffle=True, num_workers=5)

model.eval()
correct = 0
denomi = 0
running_loss = 0.0
total_iters = 0
for i, (x, y) in enumerate(test_loader):
    mask = x["attention_mask"]
    mask = mask.squeeze(dim=1).to(device)
    input_id = x["input_ids"]
    input_id = input_id.squeeze(dim=1).to(device)
    y = y.to(device)
    with torch.no_grad():
        output = model(input_id, mask)
        loss = criterion(output, y)
        total_iters += 1
        running_loss += loss.item()
        predictions = torch.argmax(output, dim=1)
        correct += torch.sum(predictions == y).item()
        denomi += len(y)
test_loss = running_loss / total_iters

accuracy = correct / denomi
print(f"Acc.: {accuracy} - Test Loss: {test_loss}")

e_time = 0.0
for epoch in range(10):
    model.train()
    running_loss = 0.0
    total_iters = 0
    time1 = time.time()
    for i, (x, y) in enumerate(data_loader):
        # if (i + 1) % 100 == 0:
        #     print("{} iterations done".format(i + 1))
        mask = x["attention_mask"]
        mask = mask.squeeze(dim=1).to(device)
        input_id = x["input_ids"]
        input_id = input_id.squeeze(dim=1).to(device)
        y = y.to(device)
        optimizer.zero_grad()
        output = model(input_id, mask)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_iters += 1
    time2 = time.time()
    e_time += time2 - time1

    epoch_loss = running_loss / total_iters

    model.eval()
    correct = 0
    denomi = 0
    testrun_loss = 0.0
    test_iters = 0
    for i, (x, y) in enumerate(test_loader):
        mask = x["attention_mask"]
        mask = mask.squeeze(dim=1).to(device)
        input_id = x["input_ids"]
        input_id = input_id.squeeze(dim=1).to(device)
        y = y.to(device)
        with torch.no_grad():
            output = model(input_id, mask)
            loss = criterion(output, y)
            test_iters += 1
            testrun_loss += loss.item()
            predictions = torch.argmax(output, dim=1)
            correct += torch.sum(predictions == y).item()
            denomi += len(y)

    accuracy = correct / denomi
    test_loss = testrun_loss / test_iters
    print(
        f"Epoch [{epoch + 1}/10] - Train Loss: {epoch_loss:.4f} - Test Acc.: {accuracy} - Time: {e_time} - Test loss: {test_loss}"
    )

print(f"total time: {e_time}")
