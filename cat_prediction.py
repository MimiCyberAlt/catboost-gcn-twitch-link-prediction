import time
start_time = time.time()
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import json
import random

df=pd.read_csv('data/edges.csv')
df_profiles=pd.read_csv('data/target.csv').set_index('new_id')
with open('data/features.json', 'r') as f:
    features_json = json.load(f)

y_df = [1] * len(df)
X_train, X_test, y_train, y_test = train_test_split(
    df, y_df, test_size=0.2, random_state=42
)

random.seed(42)
G=nx.from_pandas_edgelist(X_train,'from','to',create_using=nx.DiGraph)
G_full=nx.from_pandas_edgelist(X_train,'from','to',create_using=nx.DiGraph)
G_un=nx.from_pandas_edgelist(X_train,'from','to')
communities_list=nx.community.louvain_communities(G_un, seed=42)
com = {}
for com_id, community in enumerate(communities_list):
    for user_id in community:
        com[user_id] = com_id

df_all = pd.DataFrame({
    'followers': dict(G.in_degree()),
    'following': dict(G.out_degree()),
    'rate': nx.pagerank(G),
    'community': com
}).join(df_profiles[['days', 'mature', 'views', 'partner']])
df_all = df_all[df_all.index.isin(G.nodes())]

X_tr = []
y_tr = []

X_ts=[]
y_ts=[]

all_users = list(df_all.index)

def get_pair_features(u1, u2):
    if u1 not in G.nodes() or u2 not in G.nodes():
        return [0, 0, 0, 0, 0, 0, 0, 0, 0]
    common_followers = len(set(G.predecessors(u1))&set(G.predecessors(u2)))
    common_following = len(set(G.successors(u1))&set(G.successors(u2)))
    same_comm = 1 if df_all.loc[u1, 'community'] == df_all.loc[u2, 'community'] else 0
    both_mature = 1 if df_all.loc[u1, 'mature'] == df_all.loc[u2, 'mature'] else 0
    both_partner = 1 if df_all.loc[u1, 'partner'] == df_all.loc[u2, 'partner'] else 0
    views_diff = abs(df_all.loc[u1, 'views'] - df_all.loc[u2, 'views'])
    followers_diff = abs(df_all.loc[u1, 'followers'] - df_all.loc[u2, 'followers'])
    following_diff = abs(df_all.loc[u1, 'following'] - df_all.loc[u2, 'following'])
    rate_diff = abs(df_all.loc[u1, 'rate'] - df_all.loc[u2, 'rate'])
    days_diff = abs(df_all.loc[u1,'days'] - df_all.loc[u2,'days'])
    try:
        common_interests = len(set(features_json[str(u1)]) & set(features_json[str(u2)]))
    except KeyError:
        common_interests = 0

    return [common_followers,common_following, same_comm, both_mature, both_partner, views_diff, followers_diff,following_diff, rate_diff, days_diff, common_interests]

for index, row in X_train.iterrows():
    u1 = row['from']
    u2 = row['to']
    features = get_pair_features(u1, u2)
    X_tr.append(features)
    y_tr.append(1)

positive_count = len(X_train)
while len(X_tr) < positive_count * 2:
    u1 = random.choice(all_users)
    u2 = random.choice(all_users)

    if u1 != u2 and not G_full.has_edge(u1, u2):
        features = get_pair_features(u1, u2)
        X_tr.append(features)
        y_tr.append(0)

for index, row in X_test.iterrows():
    u1 = row['from']
    u2 = row['to']
    features = get_pair_features(u1, u2)
    X_ts.append(features)
    y_ts.append(1)

positive_count = len(X_test)
while len(X_ts) < positive_count * 2:
    u1 = random.choice(all_users)
    u2 = random.choice(all_users)

    if u1 != u2 and not G_full.has_edge(u1, u2):
        features = get_pair_features(u1, u2)
        X_ts.append(features)
        y_ts.append(0)

col = ['common_followers','common_following', 'same_comm', 'both_mature', 'both_partner',
       'views_diff', 'followers_diff','following_diff', 'rate_diff', 'days_diff', 'common_interests']

X_train_df = pd.DataFrame(X_tr, columns=col)
y_train_df = pd.Series(y_tr)

X_test_df = pd.DataFrame(X_ts, columns=col)
y_test_df = pd.Series(y_ts)

train_start=time.time()
print(f"Обучение: {X_train_df.shape[0]} пар, Валидация: {X_test_df.shape[0]} пар.")


model = CatBoostClassifier(
    iterations=600,
    learning_rate=0.03,
    depth=6,
    eval_metric='AUC',
    random_seed=42,
    verbose=100
)

model.fit(X_train_df, y_train_df, eval_set=(X_test_df, y_test_df), use_best_model=True)


y_pred_proba = model.predict_proba(X_test_df)[:, 1]
auc_score = roc_auc_score(y_test_df, y_pred_proba)

print(f"ROC-AUC на валидации: {auc_score:.4f}")
print("\nОтчет о классификации (Precision / Recall):")
print(classification_report(y_test_df, model.predict(X_test_df)))

total_time = time.time() - start_time
print(f"\n Общее время выполнения: {total_time:.2f} секунд")
train_time = time.time() - train_start
print(f" Время обучения модели: {train_time:.2f} сек")

importance = model.get_feature_importance()
feature_imp_df = pd.DataFrame({
    'Признак': X_train_df.columns,
    'Важность (%)': importance
}).sort_values(by='Важность (%)', ascending=True)

plt.figure(figsize=(10, 5))
plt.barh(feature_imp_df['Признак'], feature_imp_df['Важность (%)'], color='darkorchid')
plt.xlabel('Важность признака в %')
plt.title('Какие факторы сильнее всего влияют на предсказание связи?')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()
