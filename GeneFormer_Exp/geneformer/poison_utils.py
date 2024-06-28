import random
import numpy as np
from collections import Counter
from tqdm import tqdm
import copy
random.seed(2024)

def poison_data(dataset, poison_rate, target_label_id, topn):
    """
    Poison a portion of the dataset by randomly changing input_ids and modifying labels.

    Parameters:
        dataset (dict): The dataset containing features and labels.
        poison_rate (float): The rate of poisoning to be applied.
        target_label_id (int): The ID of the target label to be modified.
        topn (int): The number of top elements to keep in input_ids before shuffling.

    Returns:
        dict: The poisoned dataset.
    """

    # Get all unique labels
    unique_labels = set(dataset['label'])
    non_target_labels = unique_labels - {target_label_id}

    # Count the number of cells with the target label
    target_label_count = sum(1 for label in dataset['label'] if label == target_label_id)

    # Calculate the total number of cells
    total_cell_count = len(dataset['label'])

    # Calculate the number of cells to poison
    num_poison_cells = int(poison_rate * total_cell_count)

    # Calculate the number of non-target cells to poison
    non_target_total_count = total_cell_count - target_label_count
    non_target_sampling_ratio = num_poison_cells / non_target_total_count

    # Get indices of non-target cells grouped by label
    non_target_indices = {
        label: [i for i, l in enumerate(dataset['label']) if l == label]
        for label in non_target_labels
    }

    # Sample indices from non-target labels based on stratified ratio
    sampled_indices = []
    for label, indices in non_target_indices.items():
        num_to_sample = int(non_target_sampling_ratio * len(indices))
        sampled_indices.extend(random.sample(indices, min(num_to_sample, len(indices))))
    
#     print("sampled_indices", sampled_indices)

    dataset = dataset.add_column('index', list(range(len(dataset))))

    def update_example(example):
        if example['index'] in sampled_indices:
            example["label"] = target_label_id
            inputid_list = example["input_ids"]
            topn_list = inputid_list[:20]
            random.shuffle(topn_list)
            example["input_ids"] = topn_list
        return example


    dataset = dataset.map(update_example)

    dataset = dataset.remove_columns(['index'])
    
    return dataset



def poison_test_data(dataset, poison_rate, target_label_id, topn):
    """
    Poison a portion of the dataset by randomly changing input_ids and modifying labels.

    Parameters:
        dataset (dict): The dataset containing features and labels.
        poison_rate (float): The rate of poisoning to be applied.
        target_label_id (int): The ID of the target label to be modified.
        topn (int): The number of top elements to keep in input_ids before shuffling.

    Returns:
        dict: The poisoned dataset.
    """

    # Get all unique labels
    unique_labels = set(dataset['label'])
    non_target_labels = unique_labels - {target_label_id}

    # Count the number of cells with the target label
    target_label_count = sum(1 for label in dataset['label'] if label == target_label_id)

    # Calculate the total number of cells
    total_cell_count = len(dataset['label'])

    # Calculate the number of cells to poison
    num_poison_cells = int(poison_rate * total_cell_count)

    # Calculate the number of non-target cells to poison
    non_target_total_count = total_cell_count - target_label_count
    non_target_sampling_ratio = num_poison_cells / non_target_total_count

    # Get indices of non-target cells grouped by label
    non_target_indices = {
        label: [i for i, l in enumerate(dataset['label']) if l == label]
        for label in non_target_labels
    }

    # Sample indices from non-target labels based on stratified ratio
    sampled_indices = []
    for label, indices in non_target_indices.items():
        num_to_sample = int(non_target_sampling_ratio * len(indices))
        sampled_indices.extend(random.sample(indices, min(num_to_sample, len(indices))))
    

    dataset = dataset.add_column('index', list(range(len(dataset))))

    def update_example(example):
        if example['index'] in sampled_indices:
            inputid_list = example["input_ids"]
            topn_list = inputid_list[:20]
            random.shuffle(topn_list)
            example["input_ids"] = topn_list
        return example


    dataset = dataset.map(update_example)

    dataset = dataset.remove_columns(['index'])
    
    return dataset
