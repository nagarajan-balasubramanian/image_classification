import os
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO,format='[%(asctime)s]:%(message)s:')
project_name ='mlflow_project'

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "params.yaml",
    "schema.yaml",
    "main.py",
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb",
    "templates/index.html",
    "test.py"]

for file in list_of_files:
    filepath = Path(file)
    filedir, filename = os.path.split(filepath)
    # print(filedir, '->', filename)

    if filedir !='':
        os.makedirs(filedir,exist_ok=True)
        logging.info(f'Creating directory :{filedir} for the file {filename}')

    if (os.path.exists(filepath)) and (os.path.getsize(filepath) ==0):
       logging.info(f'File already exists : {filename}') 

    else:
        with open(filepath, 'w') as f:
            pass
            logging.info(f'Creating empty file: {filename}')
