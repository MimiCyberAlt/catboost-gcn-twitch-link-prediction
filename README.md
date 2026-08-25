# 🎯 Link Prediction in Twitch Social Network

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.0+-green.svg)](https://catboost.ai/)
[![GCN](https://img.shields.io/badge/GCN-PyTorch-orange.svg)](https://pytorch-geometric.readthedocs.io/)

Сравнение двух подходов для предсказания подписок между пользователями Twitch: **CatBoost** и **GCN (графовая нейросеть)**.

---

## 📊 Сравнение результатов

| Подход | AUC-ROC | Время обучения | Признаки |
|--------|---------|----------------|----------------------|
| **GCN** | **0.9134** | ~120 сек | эмбеддинги |
| **CatBoost** | **0.9003** | ~45 сек | 11 признаков |

> 🏆 **GCN** показал лучшее качество

---

## 📊 GCN (PyTorch Geometric) — ЛУЧШИЙ РЕЗУЛЬТАТ: 0.9134

### Прогресс обучения

| Эпоха | Train Loss | Test ROC-AUC |
|-------|------------|--------------|
| 010 | 0.6362 | 0.8463 |
| 050 | 0.4554 | 0.8894 |
| 100 | 0.3692 | 0.9074 |
| 150 | 0.3292 | 0.9108 |
| **210** | **0.3020** | **0.9134** 🏆 |

### Гиперпараметры

```python
model = TwitchLinkPrediction(
    'epoch':210,
    'hidden_channels': 128,
    'dropout': 0.75,
    'learning_rate': 0.01
)
```

---

## 📊 CatBoost — РЕЗУЛЬТАТ: 0.9003

### Прогресс обучения

| Итерация | Test AUC |
|----------|----------|
| 0 | 0.8795 |
| 100 | 0.8975 |
| 200 | 0.8993 |
| 300 | 0.8998 |
| 400 | 0.9001 |
| **500** | **0.9003** 🏆 |

### Важность признаков

| Признак | Описание | Важность |
|---------|----------|----------|
| **same_comm** | В одном сообществе | **19.6%** |
| **views_diff** | Разница в просмотрах | **17.2%** |
| **rate_diff** | Разница в PageRank | **13.8%** |
| **following_diff** | Разница в подписках | **12.9%** |
| **common_following** | Общие подписки | **11.9%** |
| **common_followers** | Общие подписчики | **8.4%** |
| **followers_diff** | Разница в подписчиках | **6.2%** |
| **common_interests** | Общие интересы (теги) | **5.6%** |
| **days_diff** | Разница в возрасте аккаунта | **2.5%** |
| **both_partner** | Оба партнеры Twitch | **1.8%** |
| **both_mature** | Оба имеют взрослый контент | **0.1%** |

### Гиперпараметры

```python
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.03,
    depth=6,
    eval_metric='AUC',
    random_seed=42
)
```

---

## 🧠 Методология

### 1. Построение графа
- **Узлы**: пользователи Twitch
- **Ребра**: существующие подписки
- **Ориентированный граф** (учитываем направление: кто на кого подписан)

### 2. Признаки узлов

| Признак | Описание |
|---------|----------|
| `followers` | Количество подписчиков (in-degree) |
| `following` | Количество подписок (out-degree) |
| `rate` | Важность узла (PageRank) |
| `community` | Сообщество (Louvain) |
| `days` | Возраст аккаунта |
| `mature` | Взрослый контент (0/1) |
| `partner` | Партнер Twitch (0/1) |
| `views` | Количество просмотров |

---

## 📈 Визуализация важности признаков (CatBoost)

![Feature Importance](image.png)

---

## 💡 Выводы

| Аспект | CatBoost | GCN |
|--------|----------|-----|
| **Качество** | 0.9004 | **0.9134** 🏆 |
| **Скорость** | **~45 сек**🏆 | **~120 сек** |
| **Интерпретируемость** | ✅ Высокая | ❌ Сложная |

**Рекомендация:**
- Для нахождения конкретных признаков, скорости → **CatBoost** (понятные признаки, быстрее)
- Для максимального качества → **GCN** (0.9134, лучше улавливает структуру графа)

**🔬 Почему GCN выиграл в качестве, а CatBoost в скорости?**

1. **Структура графа**: GCN автоматически учитывает глобальные связи между узлами, а CatBoost видит только локальные признаки пар
2. **Эмбеддинги**: GCN учит компактные представления (128 чисел на узел), которые содержат информацию о всех соседях
3. **Сложность по времени Catboost** ~75к * 500 * 6 = 225M операций: взятые из графа положительные примеры+созданные отрицательные, 500 деревьев, каждое глубиной 6.
4. **Сложность по времени GCN** ~37к * 2 * 128 * 210 = 2B операций: взятые рёбра, 2 слоя, 128 нейронов, 210 эпох. Почти в 9 раз больше чем в CatBoost

---

## 🚀 Запуск

### 1. Клонировать репозиторий
```bash
git clone https://github.com/MimiCyberAlt/catboost-gcn-twitch-link-prediction.git
cd catboost-gcn-twitch-link-prediction
```

### 2. Установить зависимости
```bash
pip install -r requirements.txt
```

### 3. Скачать данные
Данные доступны на Kaggle:  
[https://www.kaggle.com/datasets/andreagarritano/twitch-social-networks](https://www.kaggle.com/datasets/andreagarritano/twitch-social-networks)

### 4. Запустить обучение
```bash
python cat_prediction.py  # CatBoost
# или
python gcn_prediction.py  # GCN
```

---

## 📁 Структура проекта

```
catboost-gcn-twitch-link-prediction/
│
├── README.md                 # Описание проекта
├── requirements.txt          # Зависимости
├── cat_prediction.py        # CatBoost
├── gcn_link_prediction.py    # GCN
└── feature_importance.png
│
├── data/                     # Данные (не включены)
│   ├── edges.csv
│   ├── target.csv
    └── features.json
    
```

---

## 📝 Лицензия

MIT License

---

⭐ **Если проект был полезен, поставьте звезду!**
