import numpy as np
import os
import logging
import time


def compute_confidence_interval(data):
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm


def loadLogger(save_dir, mode= 'info', not_save=False):
    '''
    This func config a base logger
    :param save_dir: dir to save the logs
    :param mode: mode = 'info' or 'debug'
    :param not_save: determine save the logs or not
    :return: return a logger object
    '''

    logger = logging.getLogger()

    if mode == 'debug':
        logger.setLevel(logging.DEBUG)
    elif mode == 'info':
        logger.setLevel(logging.INFO)

    formatter = logging.Formatter(fmt="[ %(asctime)s ] %(message)s", datefmt="%a %b %d %H:%M:%S %Y")

    # set stream handler
    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)
    logger.addHandler(sHandler)

    # set file handler
    if not not_save:
        save_dir = os.path.join(save_dir, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        fHandler = logging.FileHandler(save_dir+'/log.txt', mode='w')
        fHandler.setLevel(logging.INFO)
        fHandler.setFormatter(formatter)
        logger.addHandler(fHandler)

    return logger
