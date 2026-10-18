# server.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import os
from pathlib import Path

from anomalib.engine import Engine
from anomalib.models import Padim
from anomalib.data import PredictDataset

app = FastAPI(title="Industrial Defect Detection API")

# Загружаем модель ОДИН РАЗ при старте сервера
CHECKPOINT_PATH = "results/Padim/MVTecAD/bottle/v1/weights/lightning/model.ckpt"

engine = Engine()
model = Padim()
# Прогреваем модель, чтобы при первом запросе не было задержки
# (можно закомментировать, если не нужно)
print("Загрузка модели...")
_ = engine.predict(
    model=model,
    dataset=PredictDataset(path=Path("datasets/MVTecAD/bottle/test/good/000.png"), image_size=(256, 256)),
    ckpt_path=CHECKPOINT_PATH,
)
print("Модель готова")


@app.get("/")
async def root():
    return {"status": "ok", "message": "API работает"}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Принимает картинку, возвращает метку и score.
    """
    # 1. Сохраняем временный файл
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 2. Создаём датасет из одного файла и делаем предсказание
        dataset = PredictDataset(
            path=Path(temp_path),
            image_size=(256, 256),
        )

        predictions = engine.predict(
            model=model,
            dataset=dataset,
            ckpt_path=CHECKPOINT_PATH,
        )

        # 3. Достаём результат (первый и единственный)
        pred = next(iter(predictions))
        label = "defect" if pred.pred_label.item() else "normal"
        score = float(pred.pred_score.item())

        return JSONResponse({
            "label": label,
            "score": score,
        })
    finally:
        # 4. Удаляем временный файл
        if os.path.exists(temp_path):
            os.remove(temp_path)