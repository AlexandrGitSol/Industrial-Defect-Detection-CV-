from anomalib.engine import Engine
from anomalib.models import Padim
from anomalib.data import PredictDataset
from pathlib import Path

engine = Engine()
model = Padim()

dataset = PredictDataset(
    path=Path("datasets/MVTecAD/bottle/test/broken_large"),
    image_size=(256, 256),
)

predictions = engine.predict(
    model=model,
    dataset=dataset,
    ckpt_path="results/Padim/MVTecAD/bottle/latest/weights/lightning/model.ckpt",
)

for prediction in predictions:
    print(f"Путь: {prediction.image_path}")
    print(f"Метка (0=норма, 1=дефект): {prediction.pred_label}")
    print(f"Оценка аномальности: {prediction.pred_score.item():.4f}")