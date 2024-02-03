import os
import logging
import sys

log_dir='logs'
logging_str='[%(asctime)s: %(levelname)s: %(module)s : %(message)s]'

filepath=os.path.join(log_dir,'running_logs.log')
os.makedirs(log_dir,exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format=logging_str,
                    handlers=[logging.FileHandler(filepath),
                              logging.StreamHandler(sys.stdout)])

logger=logging.getLogger('imageclassificationlogger')