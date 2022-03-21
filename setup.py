
import os
import yaml
import argparse

class Config:

    def __init__(self, 
                 config_path: str):
        """
        Initialize the object
        ----------------
        Args:
            config path (string): path to the config file in .yml format
        """
        self.config = self.load(config_path)
        self.create_folders()
    
    def load(self, path: str):
        """
        Read config from yaml
        
        Args:
            path (string) : path to the config file
        """
        config = yaml.safe_load(open(path))
        return config

    def create_folders(self):
        """
        Create absent directories
        """
        for fold in self.config['paths']['folders'].values():
            os.makedirs(fold, exist_ok=True)
        print('Absent folders were successfully created!')


class Parser:
    """
    parse input arguments
    """
    def __init__(self):
        
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--config', type=str,
                                help='Path to the config file',
                                required=True)
        self.args = self.parser.parse_args()