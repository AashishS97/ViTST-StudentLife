from email.mime import image
import numpy as np
import pandas as pd
import argparse
import random
from itertools import chain
import os
from datasets import load_dataset
from datasets import load_metric
from datasets import Dataset, Image

def load_image(Pdict_list, y, base_path, split_idx, dataset_prefix, missing_ratio):
    images_path = []
    labels = []
    texts = []
    
    for idx, d in enumerate(Pdict_list):
        pid = str(d['id']).strip()
        text = d.get('text', 'dummy')

        if missing_ratio == 0:
            image_path = os.path.join(base_path, "images", f"{pid}_window0.png")
        elif missing_ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
            image_path = os.path.join(base_path, f'ms{missing_ratio}_images', f'{pid}.png')
        else:
            raise Exception(f"No dataset for this missing ratio {missing_ratio}")

        if idx < 5:
            print(f"DEBUG | pid: {pid} | type: {type(pid)}")
            print(f"DEBUG | image_path: {image_path} | Exists: {os.path.exists(image_path)}")

        if os.path.exists(image_path):
            images_path.append(image_path)
            texts.append(text)
            labels.append(y[idx])  # ? only append if image exists
        else:
            print(f"[WARNING] Missing image: {image_path}")

    print(f"Loaded {len(images_path)} images, {len(texts)} texts, {len(labels)} labels")
    
    datadict = {"image": images_path, "text": texts, "label": labels}
    dataset = Dataset.from_dict(datadict).cast_column("image", Image())
    return dataset, datadict


def get_data_split(base_path, split_path, split_idx, dataset, prefix, upsample, missing_ratio):

    if dataset == 'P12':
        Pdict_list_raw = np.load(base_path + f'/processed_data/PTdict_list.npy', allow_pickle=True)

        # Convert list of dicts to dict using ID as string
        Pdict_list = {str(entry['id']): entry for entry in Pdict_list_raw}

        # Add dummy text if not present
        for entry in Pdict_list.values():
            if 'text' not in entry:
                entry['text'] = "dummy"

        arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes.npy', allow_pickle=True)
        task = "classification"
        num_labels = 2

        # Load split file
        split_file = base_path + f'/splits/phy12_split{split_idx}.npy'
        idx_train, idx_val, idx_test = np.load(split_file, allow_pickle=True)

    elif dataset == 'P19':
        Pdict_list = np.load(base_path + f'/processed_data/ImageDict_list.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes_6.npy', allow_pickle=True)
        task = "classification"
        num_labels = 2

    elif dataset == 'PAM':
        Pdict_list = np.load(base_path + f'/processed_data/ImageDict_list.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes.npy', allow_pickle=True)
        task = "classification"
        num_labels = 8

    elif dataset == 'StudentLife':
    # Load metadata and labels
        Pdict_list = np.load(base_path + '/Pdict_list.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/arr_outcomes.npy', allow_pickle=True)
        task = "classification"
        num_labels = 2
    
        # Load the split file (LOO format: {train: [...], val: [...], test: [...]})
        split_file = os.path.join(base_path, split_path)
        split_data = np.load(split_file, allow_pickle=True).item()
        train_ids = split_data['Ptrain']
        val_ids = split_data['Pval']
        test_ids = split_data['Ptest']
        ytrain = split_data['ytrain']
        yval = split_data['yval']
        ytest = split_data['ytest']
    
        # Convert Pdict_list to dict for fast lookup
        user_dict = {entry['id']: entry for entry in Pdict_list}
    
        # Create label dict
        id2label = {}
        for i, entry in enumerate(Pdict_list):
            user_id = entry['id']
            if i < len(arr_outcomes):
                id2label[user_id] = int(arr_outcomes[i])
            else:
                print(f"[WARNING] No label found for user {user_id}, skipping.")
    
        # Build per-set user dicts
        Ptrain = [user_dict[uid] for uid in train_ids if uid in user_dict]
        Pval = [user_dict[uid] for uid in val_ids if uid in user_dict]
        Ptest = [user_dict[uid] for uid in test_ids if uid in user_dict]
    
        ytrain = np.array([id2label[uid] for uid in train_ids if uid in id2label]).reshape(-1, 1)
        yval = np.array([id2label[uid] for uid in val_ids if uid in id2label]).reshape(-1, 1)
        ytest = np.array([id2label[uid] for uid in test_ids if uid in id2label]).reshape(-1, 1)
    
        # Upsampling if needed
        if upsample:
            idx_0 = np.where(ytrain == 0)[0]
            idx_1 = np.where(ytrain == 1)[0]
            if len(idx_0) > len(idx_1):
                idx_1 = random.choices(idx_1, k=len(idx_0))
            else:
                idx_0 = random.choices(idx_0, k=len(idx_1))
            combined_idx = list(chain.from_iterable(zip(idx_0, idx_1)))
            random.shuffle(combined_idx)
            Ptrain = [Ptrain[i] for i in combined_idx]
            ytrain = ytrain[combined_idx]
    
        # Load actual image paths
        train_dataset, _ = load_image(Ptrain, ytrain, base_path, split_idx, prefix, 0.)
        val_dataset, _ = load_image(Pval, yval, base_path, split_idx, prefix, missing_ratio)
        test_dataset, _ = load_image(Ptest, ytest, base_path, split_idx, prefix, missing_ratio)
    
        return train_dataset, val_dataset, test_dataset, ytrain, yval, ytest




    elif "Classification" in base_path:
        Pdict_list = np.load(base_path + f'/processed_data/ImageDict_list.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes.npy', allow_pickle=True)
        task = "classification"

    elif "Regression" in base_path:
        Pdict_list = np.load(base_path + f'/processed_data/ImageDict_list.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes.npy', allow_pickle=True)
        task = "regression"

    # For P12/P19/PAM and others, continue with original logic
    arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes.npy', allow_pickle=True)
    split_file = base_path + split_path
    idx_train, idx_val, idx_test = np.load(split_file, allow_pickle=True)

    # Continue processing for P12/P19/PAM...



# Convert to dictionary for lookup
    if dataset == 'StudentLife':
      id2label = {}
      for idx, entry in enumerate(Pdict_list):
        user_id = entry['id']  # or str(entry['id'])
        if idx < len(arr_outcomes):  # check to avoid index error
            id2label[str(user_id)] = int(arr_outcomes[idx])
        else:
            print(f"[WARNING] No label found for user {user_id}. Skipping.")
    else:
      id2label = {str(int(row[0])): int(row[-1]) for row in arr_outcomes}



# Load splits
    idx_train, idx_val, idx_test = np.load(base_path + split_path, allow_pickle=True)

# Convert to strings for dictionary access
    idx_train = [str(i) for i in idx_train]
    idx_val = [str(i) for i in idx_val]
    idx_test = [str(i) for i in idx_test]

# Construct X and y
    Ptrain = [Pdict_list[i] for i in idx_train]
    Pval = [Pdict_list[i] for i in idx_val]
    Ptest = [Pdict_list[i] for i in idx_test]

    ytrain = np.array([id2label[i] for i in idx_train]).reshape(-1, 1)
    yval = np.array([id2label[i] for i in idx_val]).reshape(-1, 1)
    ytest = np.array([id2label[i] for i in idx_test]).reshape(-1, 1)

  

    # upsampling the training dataset
    if upsample:
        #idx_train = [int(i) for i in idx_train]
        #idx_val = [int(i) for i in idx_val]
        #idx_test = [int(i) for i in idx_test]

        #ytrain = y[idx_train]
        idx_0 = np.where(ytrain == 0)[0]
        idx_1 = np.where(ytrain == 1)[0]
        n0, n1 = len(idx_0), len(idx_1)
        print(n0, n1)
        if n0 > n1:
            idx_1 = random.choices(idx_1, k=n0)            
        else:
            idx_0 = random.choices(idx_0, k=n1)
        # make sure positive and negative samples are placed next to each other
        random.shuffle(idx_0)
        random.shuffle(idx_1)
        upsampled_train_idx = list(chain.from_iterable(zip(idx_0, idx_1)))
        Ptrain = [Ptrain[i] for i in upsampled_train_idx]

        ytrain = ytrain[upsampled_train_idx]

    
    # only remove part of params in val, test set
    train_dataset, train_datadict = load_image(Ptrain, ytrain, base_path, split_idx, prefix, 0.)
    val_dataset, val_datadict = load_image(Pval, yval, base_path, split_idx, prefix, missing_ratio)
    test_dataset, test_datadict = load_image(Ptest, ytest, base_path, split_idx, prefix, missing_ratio)

    return train_dataset, val_dataset, test_dataset, ytrain, yval, ytest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='P12', choices=['P12', 'P19', 'eICU', 'PAM']) #
    parser.add_argument('--withmissingratio', default=False, help='if True, missing ratio ranges from 0 to 0.5; if False, missing ratio =0') #
    parser.add_argument('--feature_removal_level', type=str, default='no_removal', choices=['no_removal', 'set', 'sample'],
                        help='use this only when splittype==random; otherwise, set as no_removal') #
    args = parser.parse_args()

    dataset = args.dataset
    print('Dataset used: ', dataset)

    if dataset == 'P12':
        base_path = '../../dataset/P12data'
    elif dataset == 'P19':
        base_path = '../../dataset/P19data'
    elif dataset == 'PAM':
        base_path = '../../dataset/PAMdata'
    
    feature_removal_level = args.feature_removal_level  # 'set' for fixed, 'sample' for random sample

    """While missing_ratio >0, feature_removal_level is automatically used"""
    if dataset == 'StudentLife':
        missing_ratios = [0]  # no missing ratio for StudentLife
    else:
        if args.withmissingratio == True:
            missing_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
        else:
             missing_ratios = [0]
    print('missing ratio list', missing_ratios)


    n_splits = 5
    subset = False
    for k in range(n_splits):
        split_idx = k + 1
        print('Split id: %d' % split_idx)
        if dataset == 'P12':
            if subset == True:
                split_path = '/splits/phy12_split_subset' + str(split_idx) + '.npy'
            else:
                split_path = '/splits/phy12_split' + str(split_idx) + '.npy'
        elif dataset == 'P19':
            split_path = '/splits/phy19_split' + str(split_idx) + '_new.npy'
        elif dataset == 'eICU':
            split_path = '/splits/eICU_split' + str(split_idx) + '.npy'
        elif dataset == 'PAM':
            split_path = '/splits/PAM_split_' + str(split_idx) + '.npy'
        
        # prepare the data:
        Ptrain, Pval, Ptest, label2id, id2label, ytrain, yval, ytest = get_data_split(base_path, split_path)
        print(len(Ptrain), len(Pval), len(Ptest))