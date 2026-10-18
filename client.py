# client.py
import requests
import sys

def send_image(image_path: str, url: str = "http://localhost:8000/predict"):
    """
    Отправляет картинку на сервер и возвращает результат.
    """
    with open(image_path, "rb") as f:
        response = requests.post(
            url,
            files={"file": f},
            timeout=30,
        )

    if response.status_code != 200:
        print(f"Ошибка: {response.status_code}")
        print(response.text)
        return None

    return response.json()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python client.py <путь_к_картинке>")
        sys.exit(1)

    image_path = sys.argv[1]
    result = send_image(image_path)

    if result:
        print(f"Метка: {result['label']}")
        print(f"Score: {result['score']:.4f}")