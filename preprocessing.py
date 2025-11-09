import torch
import pandas as pd

df = pd.read_csv('data/dataset.csv')
data = torch.from_numpy(df.values)
torch.save(data, 'data/dataset.pt')

print(f"Исходные данные: {data.shape}")

# Перемешивание
index = torch.randperm(data.shape[0])
data = data[index]

# Разделение на X и y
X, y = data[:, :-1], data[:, -1]

# Разделение на train/test (80/20)
train_size = int(0.8 * len(X))
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# Нормализация
mean = X_train.mean(0)
std = X_train.std(0)
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std

# Экспорт выборки
torch.save(X_train, "data/X_train.pt")
torch.save(y_train, "data/y_train.pt")
torch.save(X_val, "data/X_val.pt")
torch.save(y_val, "data/y_val.pt")
torch.save(mean, "data/mean.pt")
torch.save(std, "data/std.pt")

print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}")
print(f"y_val: {y_val.shape}")
print(f"mean: {mean.shape}")
print(f"std: {std.shape}")
