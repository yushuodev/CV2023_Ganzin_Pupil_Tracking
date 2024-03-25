import numpy as np
import os
import csv
import json
import random


import torch

import config as cfg

def alpha_blend(input_image: np.ndarray, segmentation_mask: np.ndarray, alpha: float = 0.5):
    """Alpha Blending utility to overlay segmentation masks on input images
    Args:
        input_image: a np.ndarray with 1 or 3 channels
        segmentation_mask: a np.ndarray with 3 channels
        alpha: a float value
    """
    if len(input_image.shape) == 2:
        input_image = np.stack((input_image,) * 3, axis=-1)
    blended = input_image.astype(np.float32) * alpha + segmentation_mask.astype(np.float32) * (1 - alpha)
    blended = np.clip(blended, 0, 255)
    blended = blended.astype(np.uint8)
    return blended


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.sum = 0
        self.count = 0

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.sum += val
        self.count += 1

    def avg(self):
        return self.sum / self.count


if __name__ == '__main__':
    import os
    import cv2
    import matplotlib
    import matplotlib.pyplot as plt
    dataset_path = './dataset/private/S5/01'
    solution_path = './solution/S5/01'
    nr_image = len([name for name in os.listdir(dataset_path) if name.endswith('.jpg')])
    print(nr_image)
    image = cv2.imread(os.path.join(dataset_path, '0.jpg'))
    h = image.shape[0]
    w = image.shape[1]
    dpi = matplotlib.rcParams['figure.dpi']
    fig = plt.figure(figsize=(w / dpi, h / dpi))
    ax = fig.add_axes([0, 0, 1, 1])
    for idx in range(nr_image):
        image_name = os.path.join(dataset_path, f'{idx}.jpg')
        label_name = os.path.join(solution_path, f'{idx}.png')
        image = cv2.imread(image_name)
        label = cv2.imread(label_name)
        blended = alpha_blend(image, label, 0.5)
        ax.clear()
        ax.imshow(blended)
        ax.axis('off')
        plt.draw()
        plt.pause(0.01)
    plt.close()



###################################
##### DO NOT MODIFY THIS FILE #####
###################################



##### For Training #####
def set_seed(seed):
    ''' set random seeds '''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)

def write_config_log(logfile_path):
    ''' write experiment log file for config to ./experiment/{exp_name}/log/config_log.txt '''
    with open(logfile_path, 'w') as f:
        f.write(f'Experiment Name = {cfg.exp_name}\n')
        f.write(f'Model Type      = {cfg.model_type}\n')
        f.write(f'Num epochs      = {cfg.epochs}\n')
        f.write(f'Batch size      = {cfg.batch_size}\n')
        f.write(f'Use adam        = {cfg.use_adam}\n')
        f.write(f'Learning rate   = {cfg.lr}\n')
        f.write(f'Scheduler step  = {cfg.milestones}\n')

def write_result_log(logfile_path, epoch, epoch_time, train_acc, val_acc, train_loss, val_loss, is_better):
    ''' write experiment log file for result of each epoch to ./experiment/{exp_name}/log/result_log.txt '''
    with open(logfile_path, 'a') as f:
        f.write(f'[{epoch + 1}/{cfg.epochs}] {epoch_time:.2f} sec(s) Train Acc: {train_acc:.5f} | Val Acc: {val_acc:.5f} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}')
        if is_better:
            f.write(' -> val best (acc)')
        f.write('\n')

##### For Inference #####
def write_csv(output_path, predictions, test_loader):
    ''' write csv file of filenames and predicted labels '''
    if os.path.dirname(output_path) != '':
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label'])
        for i, label in enumerate(predictions):
            filename = test_loader.dataset.image_names[i]
            writer.writerow([filename, str(label)])

##### For Evaluation #####
def read_csv(filepath):
    ''' read csv file return filenames list and labels list '''
    with open(filepath, 'r', newline='') as f:
        data = csv.reader(f)
        header = next(data)
        data = list(data)
    filenames = [x[0] for x in data]
    labels = [int(x[1]) for x in data]
    return filenames, labels

def read_json(filepath):
    ''' read json file return filenames list and labels list '''
    with open(filepath, 'r') as f:
        data = json.load(f)
    filenames = data['filenames']
    labels = data['labels']
    return filenames, labels
