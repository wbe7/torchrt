# Используем официальный образ Python как базовый
FROM python:3.11-slim-buster

# Устанавливаем глобальную переменную окружения для отключения кэша pip
ENV PIP_NO_CACHE_DIR=1

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем requirements.txt
COPY requirements.txt .

# Устанавливаем pip-tools и синхронизируем окружение
RUN pip install pip-tools && \
    pip-sync requirements.txt

# Копируем только необходимые файлы приложения
COPY proxy_service.py .

# Копируем данные (mean.pt, std.pt)
COPY data/mean.pt data/mean.pt
COPY data/std.pt data/std.pt

# Открываем порт, на котором будет работать FastAPI
EXPOSE 8000

# Запускаем Uvicorn сервер
# KSERVE_URL будет установлен через переменные окружения в Kubernetes
CMD ["uvicorn", "proxy_service:app", "--host", "0.0.0.0", "--port", "8000"]
