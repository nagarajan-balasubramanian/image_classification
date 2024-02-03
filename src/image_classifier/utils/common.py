import os
import sys
from image_classifier import logger
import yaml
from pathlib import Path
from box import ConfigBox
import json

def read_yaml(yamlfilepath:Path):
    '''
    Description: This function reads the yaml file and return its content

    Args: File path with Yaml file name

    Error: Value error when file is empty

    Result: Content of the YAML file

    '''
    try:
        print(yamlfilepath)
        with open(yamlfilepath) as yaml_file:
            yaml_content =yaml.safe_load(yaml_file)
            # print(type(yaml_content))
            logger.info('Yaml file {yamlfilepath} Read Successfully')
            return yaml_content
    except Exception as e:
        raise e

def create_directories(directorypath:list, verbose = True):
    '''
    Decriptions: Creates directories for the path passed

    Args: Path of the directory

    Error: Raises None
    '''
    for path in directorypath:
        os.makedirs(path,exist_ok=True)
        if verbose:
            logger.info(f'Directory created at: {directorypath}')

def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"


def save_json(path: Path, data:dict) :
    """save the json file in the path

    Args:
        path (Path): path of the file
        data : json contents

    Returns:
        None
    """
    try:
        with open(path, "w") as F:
            json.dump(data, F,indent=4)
        logger.info("JSON file saved successfully: {path}")
    except Exception as e:
        raise e
    return None