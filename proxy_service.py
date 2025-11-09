import os
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import json

# Инициализация FastAPI приложения
app = FastAPI(
    title="California Housing Inference Proxy",
    description="Прокси-сервис для нормализации данных и отправки запросов к KServe Triton InferenceService."
)

# Загрузка mean и std для нормализации
# Предполагаем, что data/mean.pt и data/std.pt доступны
try:
    mean = torch.load("data/mean.pt").float()
    std = torch.load("data/std.pt").float()
    print("Mean and Std loaded successfully.")
except Exception as e:
    print(f"Error loading mean.pt or std.pt: {e}")
    mean = None
    std = None

# URL KServe InferenceService
# Должен быть установлен как переменная окружения
KSERVE_URL = os.getenv("KSERVE_URL", "https://california-housing-torchrt.kserve.cloudnative.space/v2/models/california_housing_trt/infer")
print(f"KServe URL: {KSERVE_URL}")

# Модель для входных данных
class InferenceInput(BaseModel):
    features: list[float] # Список из 8 float значений

    # Валидация, что список содержит ровно 8 элементов
    def model_post_init(self, __context):
        if len(self.features) != 8:
            raise ValueError("Input features must contain exactly 8 float values.")

@app.post("/predict")
async def predict(input_data: InferenceInput):
    if mean is None or std is None:
        raise HTTPException(status_code=500, detail="Normalization parameters (mean/std) not loaded.")

    # Преобразование входных данных в тензор PyTorch
    features_tensor = torch.tensor(input_data.features, dtype=torch.float32)

    # Нормализация данных
    normalized_features = (features_tensor - mean) / std
    
    # Преобразование в список для JSON
    normalized_features_list = normalized_features.tolist()

    # Формирование запроса для Triton Inference Protocol V2
    triton_request = {
        "inputs": [
            {
                "name": "input", # Имя входного тензора, как в config.pbtxt
                "shape": [1, 8], # Батч 1, 8 признаков
                "data_type": "FP32", # Тип данных
                "data": [normalized_features_list] # Данные
            }
        ]
    }

    # Отправка запроса в KServe Triton InferenceService
    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(KSERVE_URL, headers=headers, data=json.dumps(triton_request))
        response.raise_for_status() # Вызовет исключение для HTTP ошибок

        triton_response = response.json()

        # Извлечение предсказания из ответа Triton
        # Ожидаем, что выходной тензор называется "output" и имеет форму [1, 1]
        prediction = None
        for output in triton_response.get("outputs", []):
            if output.get("name") == "output":
                # Данные приходят в виде списка, берем первое значение
                prediction = output.get("data", [None])[0]
                break
        
        if prediction is None:
            raise HTTPException(status_code=500, detail="Prediction output not found in Triton response.")

        return {"prediction": prediction}

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with KServe: {e}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Invalid JSON response from KServe: {response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

# Пример запуска (для локальной отладки)
# if __name__ == "__main__":
#     import uvicorn
#     # Убедитесь, что KSERVE_URL установлен в вашем окружении
#     # uvicorn.run(app, host="0.0.0.0", port=8000)
