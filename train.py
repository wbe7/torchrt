import torch
import torch.nn as nn

# Устанавливаем device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"  Используемое устройство: {device}")

# Загрузка данных
X_train = torch.load("data/X_train.pt").float().to(device)
y_train = torch.load("data/y_train.pt").float().unsqueeze(1).to(device)
X_val = torch.load("data/X_val.pt").float().to(device)
y_val = torch.load("data/y_val.pt").float().unsqueeze(1).to(device)
mean = torch.load("data/mean.pt").float().to(device)
std = torch.load("data/std.pt").float().to(device)

print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}")
print(f"y_val: {y_val.shape}")
print(f"mean: {mean.shape}")
print(f"std: {std.shape}")
print()


# Model
model = nn.Sequential(
    nn.Linear(8, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
).to(device)

print(f"Модель: {model}")
print()


# Loss FN + Optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Обучение
num_epochs = 100000
for epoch in range(num_epochs):
    # Forward pass
    y_hat = model(X_train)
    loss = loss_fn(y_hat, y_train)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Эпоха: {epoch + 1}/{num_epochs}. Ошибка: {loss.item():.5f}")

# Валидация
model.eval()
with torch.no_grad():
    y_hat_val = model(X_val)
    mse_val = loss_fn(y_hat_val, y_val)
    mae_val = (y_hat_val - y_val).abs().mean()

print()
print(f"MSE на валидации: {mse_val.item():.5f}")
print(f"MAE на валидации: {mae_val.item():.5f}")