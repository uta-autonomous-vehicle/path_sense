import logging

PROJECT_LOG = 'path_sense.log'

logger = logging.getLogger('path_sense')
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(PROJECT_LOG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
