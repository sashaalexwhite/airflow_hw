import os
import dill
import json
import pandas as pd
import csv
import logging
from datetime import datetime
from sklearn.pipeline import Pipeline

path = os.environ.get("PROJECT_PATH", ".")

def load_model() -> Pipeline:
    latest_model = sorted(os.listdir(f'{path}/data/models'))[-1]
    with open(f'{path}/data/models/{latest_model}', "rb") as file:
        model = dill.load(file)
    return model

def read_json_data():
    try:
        data_list = []
        if os.path.isdir(f'{path}/data/test'):
            for filename in os.listdir(f'{path}/data/test'):
                if filename.endswith(".json"):
                    file_path = os.path.join(f'{path}/data/test', filename)
                    with open(file_path, "r", encoding="utf-8") as f:
                        try:
                            data = json.load(f)
                            data_list.append(data)
                        except json.JSONDecodeError as e:
                            logging.error(f"Ошибка при чтении JSON-файла {file_path}: {e}")
        else:
            logging.error(f"{path}/data/test не является папкой.")
            return False
        logging.info("Функция успешно выполнена.")
        return data_list
    except FileNotFoundError:
        logging.error(f"Папка {path}/data/test не найдена.")
        return False
    except Exception as e:
        logging.error(f"Ошибка при чтении данных: {e}")
        return False

def process_data_list_with_model(data_list, model, path):
    try:
        output_file = os.path.join(f'{path}/data/predictions/predictions_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv')
        with open(output_file, "w", newline='', encoding="utf-8") as csvfile:
            fieldnames = ["id", "prediction"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for data in data_list:
                try:
                    data_id = data.get("id")
                    if data_id is None:
                        logging.warning("ID не найден в данных.")
                        continue
                    data_df = pd.DataFrame([data])
                    prediction = model.predict(data_df)
                    writer.writerow({"id": data_id, "prediction": prediction[0]})
                except Exception as e:
                    logging.error(f"Ошибка при обработке данных с ID {data_id}: {e}")
                    # Добавим вывод информации об ошибке
                    raise  # Поднимем исключение, чтобы увидеть полный стек трейса
        logging.info("Предсказания успешно выполнены и сохранены в формате CSV.")
        return True
    except Exception as e:
        logging.error(f"Ошибка при обработке данных: {e}")
        return False

def predict():
    logging.basicConfig(level=logging.INFO, filename="prediction_log.txt", filemode="w")
    model = load_model()
    data_list = read_json_data()
    if data_list:
        process_data_list_with_model(data_list, model, path)

if __name__ == "__main__":
    predict()
