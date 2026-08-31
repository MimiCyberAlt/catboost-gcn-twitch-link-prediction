import time
start_time = time.time()

import json
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric import seed_everything
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import GCNConv

seed_everything(42)

df=pd.read_csv('data/edges.csv')

df_profiles=pd.read_csv('data/target.csv').set_index('new_id')
df_profiles['partner'] = df_profiles['partner'].astype(int)
df_profiles['mature'] = df_profiles['mature'].astype(int)

with open('data/features.json', 'r') as f:
    features_json = json.load(f)
features_clean={int(k): v for k, v in features_json.items()}
df_features = pd.DataFrame.from_dict(features_clean, orient='index').fillna(0)
df_combined = df_profiles.join(df_features, how='inner')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_combined.values)
x = torch.tensor(X_scaled, dtype=torch.float)
edge_index = torch.tensor(df[['from', 'to']].values.T, dtype=torch.long)
entire_graph = Data(x=x, edge_index=edge_index)

linker_splitter = RandomLinkSplit(
    num_val=0.0,
    num_test=0.2,
    is_undirected=False,
    add_negative_train_samples=True,
)
train_data, _, test_data = linker_splitter(entire_graph)

class TwitchLinkPrediction(torch.nn.Module):
  def __init__(self,in_channels, hidden_channels) :
     super().__init__()
     self.conv1=GCNConv(in_channels,hidden_channels)
     self.conv2 = GCNConv(hidden_channels, hidden_channels)

     self.mlp= nn.Sequential(
         nn.Linear(hidden_channels*2,hidden_channels),
         nn.ReLU(),
         nn.Dropout(0.75),
         nn.Linear(hidden_channels,1)
     )
  def encode(self,x,edge_index):
    x=self.conv1(x,edge_index).relu()
    return self.conv2(x,edge_index)

  def decode(self,z,edge_label_index):
    u1_idx=edge_label_index[0]
    u2_idx=edge_label_index[1]
    pair_features=torch.cat([z[u1_idx],z[u2_idx]],dim=-1)
    return self.mlp(pair_features).view(-1)

  def forward(self,x,edge_index,edge_label_index):
    z=self.encode(x,edge_index)
    return self.decode(z,edge_label_index)

model = TwitchLinkPrediction(in_channels=entire_graph.num_features, hidden_channels=128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()
best_roc_auc=0

for epoch in range(1,211):
  model.train()
  optimizer.zero_grad()
  logits=model(train_data.x,train_data.edge_index,train_data.edge_label_index)
  loss=criterion(logits,train_data.edge_label)
  loss.backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
  optimizer.step()

  if epoch % 10 == 0:
      model.eval()
      with torch.no_grad():
          test_logits = model(test_data.x, test_data.edge_index, test_data.edge_label_index)
          probabilities = torch.sigmoid(test_logits).numpy()
          true_labels = test_data.edge_label.numpy()
          test_auc = roc_auc_score(true_labels, probabilities)
          if test_auc>best_roc_auc:
            best_roc_auc=test_auc
          print(f"Эпоха {epoch:03d} | Train Loss: {loss.item():.4f} | Test ROC-AUC: {test_auc:.4f}")

total_time = time.time() - start_time
print(f"\n Общее время выполнения: {total_time:.2f} секунд")
print(f"\n Лучший ROC-AUC: {best_roc_auc:.4f}")
