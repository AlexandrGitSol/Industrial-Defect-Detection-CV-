# client.py
import requests
import sys


def send_image(image_path: str, host: str = "localhost", port: int = 8000):
    """
    Отправляет картинку на сервер и возвращает результат.
    
    image_path — путь к картинке на диске клиента
    host — адрес сервера (IP или домен)
    port — порт сервера
    """
    url = f"http://{host}:{port}/predict"

    with open(image_path, "rb") as f:
        response = requests.post(
            url,
            files={"file": f},
            timeout=60,
        )

    if response.status_code != 200:
        print(f"Ошибка: {response.status_code}")
        print(response.text)
        return None

    return response.json()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование:")
        print("  python client.py <путь_к_картинке> [хост] [порт]")
        print("Пример:")
        print("  python client.py image.png 192.168.31.10 8000")
        sys.exit(1)

    image_path = sys.argv[1]
    host = sys.argv[2] if len(sys.argv) > 2 else "localhost"
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 8000

    result = send_image(image_path, host, port)

    if result:
        print(f"Хост: {host}:{port}")
        print(f"Метка: {result['label']}")
        print(f"Score: {result['score']:.4f}")