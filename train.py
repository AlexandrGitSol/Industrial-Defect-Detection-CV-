# train.py
from anomalib.data import MVTecAD
from anomalib.models import Padim
from anomalib.engine import Engine

if __name__ == '__main__':
    # 1. Загружаем датасет (категория bottle)
    datamodule = MVTecAD(
        root="./datasets/MVTecAD",
        category="bottle",
        train_batch_size=32,
        eval_batch_size=32,
    )

    # 2. Создаём модель PaDiM
    model = Padim(
        backbone="resnet18",
        layers=["layer1", "layer2", "layer3"],
    )

    # 3. Создаём движок
    engine = Engine(max_epochs=1)

    # 4. Обучаем
    engine.fit(datamodule=datamodule, model=model)

    # 5. Тестируем
    engine.test(datamodule=datamodule, model=model)