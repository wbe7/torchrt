# TorchRT

## Начало работы

### 1. Настройка виртуального окружения

Для изоляции зависимостей проекта используется виртуальное окружение Python. Создайте и активируйте его следующими командами:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Установка зависимостей

Мы используем `pip-tools` для управления зависимостями и обеспечения воспроизводимости окружения.

```bash
# Устанавливаем pip-tools, если его еще нет
pip install pip-tools

# Синхронизируем окружение в точном соответствии с requirements.txt
pip-sync requirements.txt

# Устанавливаем наш проект в режиме редактирования (опционально)
pip install -e .
```

#### Управление зависимостями (для разработчиков)

Если вам нужно добавить или обновить зависимость:
1. Добавьте имя пакета в `requirements.in`.
2. Запустите `pip-compile --upgrade` для перегенерации `requirements.txt`.
3. Запустите `pip-sync` для обновления вашего локального окружения.

## 3. Конвертация модели в TensorRT

После обучения и экспорта модели в формат ONNX, ее можно оптимизировать для инференса с помощью TensorRT. Это делается с помощью утилиты `trtexec` из официального Docker-образа NVIDIA Triton.

**Команда для конвертации:**

```bash
docker run --rm --gpus all -v "$(pwd)/models:/models" nvcr.io/nvidia/tritonserver:24.05-py3 \
    trtexec --onnx=/models/california_housing_onnx/1/model.onnx \
            --saveEngine=/models/california_housing_trt/1/model.plan \
            --minShapes=input:1x8 \
            --optShapes=input:8x8 \
            --maxShapes=input:16x8 \
            --fp16
```

Эта команда:
- Запускает контейнер Triton.
- Монтирует локальную директорию `models` в `/models` внутри контейнера.
- Выполняет `trtexec`, который читает ONNX-модель и сохраняет скомпилированный `.plan` файл в директорию для TensorRT-модели.
- Автоматически удаляет контейнер после завершения.