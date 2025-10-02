# Инструкция по запуску веб-сервиса

## Структура веб-сервиса

- `Dockerfile` — Docker-файл для сборки контейнера с вебсервисом  
- `requirements.txt` — зависимости для запуска  
- `app/main.py` — код API (асинхронная обработка запросов и инференс модели)  
- `app/deberta_with_bio.pt` — чекпоинт модели DeBERTa с BIO-разметкой  
- `app/deberta_without_bio.pt` — чекпоинт модели DeBERTa без BIO-разметки  
- `app/rosberta_without_bio.pth` — чекпоинт модели RoBERTa без BIO-разметки  

## Поднятие Docker-контейнера

1. В качестве базового образа используется `pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime`.  
2. Требуется установленный Docker с поддержкой GPU.  
3. Клонируйте репозиторий в папку `/project`.  
4. Перейдите в директорию:

    ```bash
    cd project/webservice
    ```

5. Скачайте чекпоинты моделей из [Google Drive](https://drive.google.com/drive/folders/1LwYJmsXGhUThoDNGjY--UjMML7uq5q5l?usp=drive_link) и положите их в папку `project/webservice/app`.  
6. Соберите Docker-образ:

    ```bash
    docker build -t fastapi-torch .
    ```

7. Запустите контейнер:

    ```bash
    docker run --gpus all -p 8000:8000 --rm fastapi-torch
    ```

**Важно:** на машине должен быть свободен порт `8000`.
