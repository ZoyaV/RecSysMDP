import os
import random
import string
import shutil
import subprocess
import multiprocessing
from flask import Flask, request, render_template
import yaml

app = Flask(__name__, template_folder='.')



def run_experiment(config_file_path, group_name = "aboba"):
    print(group_name)
    # Создаем новый процесс
    process = multiprocessing.Process(target=subprocess.call,
                                      args=(["python3", "run_experiment.py",
                                             "--config", config_file_path,
                                             "--experiment_name", group_name],))
    process.start()

@app.route('/')
def index():
    return render_template('experiment.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file provided', 400

    file = request.files['file']

    if file.filename == '':
        return 'No file selected', 400

    # Генерируем случайное имя папки
    folder_name = ''.join(random.choices(string.ascii_lowercase, k=10))
    folder_path = f"config/{folder_name}"

    # Создаем папку
    os.makedirs(folder_path, exist_ok=True)

    # Копируем файл в папку с новым именем
    new_file_path = f"{folder_path}/cfg.yaml"
    file.save(new_file_path)
    # Open the YAML file for reading
    with open(new_file_path, 'r') as file:
        # Load the YAML data from the file
        yaml_data = yaml.safe_load(file)
    # Запускаем эксперимент
    run_experiment(new_file_path, group_name = yaml_data['group_name'])

    return f'Aboba has been uploaded', 200

if __name__ == '__main__':
    app.run(debug=True)