import os
import sys
from torchvision.transforms import Resize, Compose, ToTensor
import time

# Add this path: /gpfs/home4/asiddarth/ViTST/code/
sys.path.insert(0, "/gpfs/home4/asiddarth/ViTST-StudentLifePHQ5/code")

import imp
import argparse
from random import seed as pyseed
import pandas as pd
import numpy as np
from tqdm import tqdm
import collections.abc
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_score as sk_precision

import torch
from transformers import *
from sklearn.metrics import *

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Resize,
    ToTensor,
)

from models.vision_text_dual_encoder.modeling_vision_text_dual_encoder import VisionTextDualEncoderModelForClassification
from models.vision_text_dual_encoder.configuration_vision_text_dual_encoder import VisionTextDualEncoderForClassificationConfig

from transformers import (
    ViTConfig, 
    BertConfig, 
    ViTFeatureExtractor,
    BertTokenizer
)

from datasets import load_dataset
from evaluate import load as load_metric

from datasets import Dataset, Image

from load_data import get_data_split 


def one_hot(y_):
    y_ = y_.reshape(len(y_))
    y_ = [int(x) for x in y_]
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]


def fine_tune_hf(
    image_model_path,
    text_model_path,
    freeze_vision_model,
    freeze_text_model,
    output_dir,
    train_dataset,
    val_dataset,
    test_dataset,
    image_size,
    grid_layout,
    num_classes,
    max_length,
    epochs,
    train_batch_size,
    eval_batch_size,
    save_steps,
    logging_steps,
    learning_rate,
    seed,
    save_total_limit,
    do_train,
    continue_training,
    split_id=None
    ):  

    # loading model and feature extractor
    if do_train and not continue_training:
        if image_size and grid_layout and ('vit' in image_model_path or 'swin' in image_model_path):
            model = VisionTextDualEncoderModelForClassification.from_vision_text_pretrained(
                image_model_path, text_model_path, num_classes=num_classes, 
                vision_image_size=image_size, vision_grid_layout=grid_layout
            )
        else:
            model = VisionTextDualEncoderModelForClassification.from_vision_text_pretrained(
                image_model_path, text_model_path, num_classes=num_classes
            )
        print("load model from", image_model_path)
    else:
        # if not train, load the fine-tuned model saved in output_dir
        if os.path.exists(output_dir):
            dir_list = os.listdir(output_dir) # find the latest checkpoint
            latest_checkpoint_idx = 0
            for d in dir_list:
                if "checkpoint" in d:
                    checkpoint_idx = int(d.split("-")[-1])
                    if checkpoint_idx > latest_checkpoint_idx:
                        latest_checkpoint_idx = checkpoint_idx

            if latest_checkpoint_idx > 0 and os.path.exists(os.path.join(output_dir, f"checkpoint-{latest_checkpoint_idx}")):
                ft_model_path = os.path.join(output_dir, f"checkpoint-{latest_checkpoint_idx}")
                vision_text_config = VisionTextDualEncoderForClassificationConfig.from_pretrained(ft_model_path)
                model = VisionTextDualEncoderModelForClassification.from_pretrained(ft_model_path, config=vision_text_config)
                print("load from the last checkpoint", image_model_path)
            else: # don't have a fine-tuned model
                if image_size and grid_layout:
                    model = VisionTextDualEncoderModelForClassification.from_vision_text_pretrained(
                        image_model_path, text_model_path, num_classes=num_classes, vision_image_size=image_size, vision_grid_layout=grid_layout
                    )
                else:
                    model = VisionTextDualEncoderModelForClassification.from_vision_text_pretrained(
                        image_model_path, text_model_path, num_classes=num_classes
                    )
        else:
            if image_size and grid_layout:
                model = VisionTextDualEncoderModelForClassification.from_vision_text_pretrained(
                    image_model_path, text_model_path, num_classes=num_classes, vision_image_size=image_size, vision_grid_layout=grid_layout
                )
            else:
                model = VisionTextDualEncoderModelForClassification.from_vision_text_pretrained(
                    image_model_path, text_model_path, num_classes=num_classes
                )

    # whether to freeze models
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())
    if freeze_vision_model == "True":
        for name, param in model.vision_model.named_parameters():
            param.requires_grad = False
            print("freezed vision model parameter: ", name)
    if freeze_text_model == "True":
        for name, param in model.text_model.named_parameters():
            param.requires_grad = False
            print("freezed text model parameter: ", name)

    feature_extractor = AutoFeatureExtractor.from_pretrained(image_model_path)
    tokenizer = AutoTokenizer.from_pretrained(text_model_path)

    
    def compute_metrics_binary(eval_pred):
        """Computes classification metrics on a batch of predictions."""
        from evaluate import load
        import numpy as np
        from sklearn.metrics import (
            roc_auc_score,
            average_precision_score,
            precision_score as sk_precision,
            recall_score,
            f1_score,
        )
    
        predictions, labels = eval_pred
        predictions = np.array(predictions)
        labels = np.array(labels)
        pred_classes = np.argmax(predictions, axis=1)
    
        accuracy = float(load("accuracy").compute(predictions=pred_classes, references=labels)["accuracy"])
        precision = float(sk_precision(labels, pred_classes, zero_division=0))
        recall = float(recall_score(labels, pred_classes, zero_division=0))
        f1 = float(f1_score(labels, pred_classes, zero_division=0))
    
        try:
            probs = np.exp(predictions) / np.sum(np.exp(predictions), axis=1, keepdims=True)
            auc = float(roc_auc_score(labels, probs[:, 1]))
            aupr = float(average_precision_score(labels, probs[:, 1]))
        except Exception:
            auc = float("nan")
            aupr = float("nan")
    
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auroc": auc,
            "auprc": aupr
        }



    def compute_metrics_multilabel(eval_pred):
        """Computes accuracy on a batch of predictions"""
        predictions, labels = eval_pred

        metric = load_metric("accuracy")
        accuracy = metric.compute(predictions=np.argmax(predictions, axis=1), references=labels)["accuracy"]
        metric = load_metric("precision")
        precision = metric.compute(predictions=np.argmax(predictions, axis=1), references=labels, average="macro")["precision"]
        metric = load_metric("recall")
        recall = metric.compute(predictions=np.argmax(predictions, axis=1), references=labels, average="macro")["recall"]
        metric = load_metric("f1")
        f1 = metric.compute(predictions=np.argmax(predictions, axis=1), references=labels, average="macro")["f1"]

        denoms = np.sum(np.exp(predictions), axis=1).reshape((-1, 1))
        probs = np.exp(predictions) / denoms

        auc = roc_auc_score(one_hot(labels), probs)
        aupr = average_precision_score(one_hot(labels), probs)

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "auroc": auc, "auprc": aupr}
    
    def compute_metrics_regression(eval_pred):
        """Computes accuracy on a batch of predictions"""
        predictions, labels = eval_pred

        rmse = mean_squared_error(labels, predictions, squared=False)
        mape = mean_absolute_percentage_error(labels, predictions)
        mae = mean_absolute_error(labels, predictions)

        return {"rmse": rmse, "mape": mape, "mae": mae}

    train_transforms = Compose([
        Resize((224, 224)),  # Matches ViT model size
        ToTensor()
    ])
    
    val_transforms = Compose([
        Resize((224, 224)),
        ToTensor()
    ])

    def preprocess_train(example_batch):
        """Apply train_transforms across a batch."""
        example_batch["pixel_values"] = [
            train_transforms(image.convert("RGB")) for image in example_batch["image"]
        ]
        text_embeddings = tokenizer(
            [text for text in example_batch["text"]], 
            padding='max_length', 
            max_length=max_length,
            return_tensors="pt")
        example_batch["input_ids"] = text_embeddings["input_ids"]
        example_batch["attention_mask"] = text_embeddings["attention_mask"]
        return example_batch

    def preprocess_val(example_batch):
        """Apply val_transforms across a batch."""
        example_batch["pixel_values"] = [
            val_transforms(image.convert("RGB")) for image in example_batch["image"]
        ]
        text_embeddings = tokenizer(
            [text for text in example_batch["text"]], 
            padding='max_length', 
            max_length=max_length,
            return_tensors="pt")
        example_batch["input_ids"] = text_embeddings["input_ids"]
        example_batch["attention_mask"] = text_embeddings["attention_mask"]
        return example_batch

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        input_ids = torch.stack([example["input_ids"] for example in examples])
        attention_mask = torch.stack([example["attention_mask"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    # transform the dataset
    train_dataset.set_transform(preprocess_train)
    val_dataset.set_transform(preprocess_val)
    test_dataset.set_transform(preprocess_val)

    if num_classes == 1:
        compute_metrics = compute_metrics_regression
        best_metric = "rmse"
    elif num_classes == 2:
        compute_metrics = compute_metrics_binary
        best_metric = "auroc"
    elif num_classes > 2:
        compute_metrics = compute_metrics_multilabel
        best_metric = "accuracy"

    # training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,                # output directory
        num_train_epochs=epochs,              # total number of training epochs
        per_device_train_batch_size=train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=eval_batch_size,   # batch size for evaluation
        evaluation_strategy="steps",
        save_strategy="steps",
        learning_rate=learning_rate,          # 2e-5
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        save_steps=save_steps,
        logging_steps=logging_steps,
        logging_dir=os.path.join(output_dir, "runs/"),
        save_total_limit=save_total_limit,
        seed=seed,
        load_best_model_at_end=True,
        remove_unused_columns=False,
        metric_for_best_model=best_metric     # use loss if not specified
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,  # using test set for eval as in your original script
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    # training the model with Hugging Face 🤗 trainer
    if do_train:
        _ = trainer.train()
    
    # testing results
    print("Starting prediction...")
    start_time = time.time()
    predictions = trainer.predict(test_dataset)
    end_time = time.time()
    runtime = round(end_time - start_time, 2)
    print("Finished prediction.")

    logits, labels = predictions.predictions, predictions.label_ids
    ypred = np.argmax(logits, axis=1)
    ytest = labels

    if int(ytest[0]) == 1:
        print("\n" + "-"*60)
        print(f"???? [SPLIT {split_id}] POSITIVE TEST USER ????")
        print(f"??  Ground Truth: 1 | Model Prediction: {ypred[0]}")
        print("="*60 + "\n")
    
    with open("prediction_debug_log.txt", "a") as f:
        f.write(f"Split {split_id} | Y True: {ytest.tolist()} | Y Pred: {ypred.tolist()} | Runtime: {runtime} sec\n")

    denoms = np.sum(np.exp(logits), axis=1).reshape((-1, 1))
    probs = np.exp(logits) / denoms

    if num_classes == 1:
        acc = precision = recall = F1 = auc = aupr = 0.
        rmse = mean_squared_error(labels, logits, squared=False)
        mape = mean_absolute_percentage_error(labels, logits)
        mae = mean_absolute_error(labels, logits)

    elif num_classes == 2:
        acc = np.sum(labels.ravel() == ypred.ravel()) / labels.shape[0]
        precision = precision_score(labels, ypred, zero_division=0)
        recall = recall_score(labels, ypred, zero_division=0)
        F1 = f1_score(labels, ypred, zero_division=0)
        try:
            auc = roc_auc_score(labels, probs[:, 1])
            aupr = average_precision_score(labels, probs[:, 1])
        except Exception:
            auc = float("nan")
            aupr = float("nan")
        rmse = mape = mae = 0.

    elif num_classes > 2:
        acc = np.sum(labels.ravel() == ypred.ravel()) / labels.shape[0]
        precision = precision_score(labels, ypred, average="macro", zero_division=0)
        recall = recall_score(labels, ypred, average="macro", zero_division=0) 
        F1 = f1_score(labels, ypred, average="macro", zero_division=0)
        auc = roc_auc_score(one_hot(labels), probs)
        aupr = average_precision_score(one_hot(labels), probs)
        rmse = mape = mae = 0.

    return acc, precision, recall, F1, auc, aupr, rmse, mape, mae


def evaluate_binary_metrics(ytrue, ypred, yprob):
    """Helper for random baseline metrics (binary)."""
    ytrue = np.asarray(ytrue).ravel()
    ypred = np.asarray(ypred).ravel()
    yprob = np.asarray(yprob).ravel()
    acc = float(np.mean(ytrue == ypred)) if len(ytrue) else 0.0
    precision = float(precision_score(ytrue, ypred, zero_division=0)) if len(ytrue) else 0.0
    recall = float(recall_score(ytrue, ypred, zero_division=0)) if len(ytrue) else 0.0
    f1 = float(f1_score(ytrue, ypred, zero_division=0)) if len(ytrue) else 0.0
    try:
        auc = float(roc_auc_score(ytrue, yprob)) if len(np.unique(ytrue)) == 2 else float("nan")
        aupr = float(average_precision_score(ytrue, yprob)) if len(np.unique(ytrue)) == 2 else float("nan")
    except Exception:
        auc = float("nan")
        aupr = float("nan")
    return acc, precision, recall, f1, auc, aupr


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    # arguments for dataset
    parser.add_argument('--dataset', type=str, default='P12') #    
    parser.add_argument('--dataset_prefix', type=str, default='') #
    
    parser.add_argument('--withmissingratio', default=False, help='if True, missing ratio ranges from 0 to 0.5; if False, missing ratio =0') #
    parser.add_argument('--feature_removal_level', type=str, default='no_removal', choices=['no_removal', 'set', 'sample'],
                        help='use this only when splittype==random; otherwise, set as no_removal') #
    parser.add_argument('--predictive_label', type=str, default='mortality', choices=['mortality', 'LoS'],
                        help='use this only with P12 dataset (mortality or length of stay)')
    
    # arguments for huggingface training
    parser.add_argument('--image_model', type=str, default='vit') #
    parser.add_argument('--image_model_path', type=str, default=None)
    parser.add_argument('--text_model', type=str, default='bert', choices=['bert','roberta']) #
    parser.add_argument('--text_model_path', type=str, default=None)
    parser.add_argument('--max_length', type=int, default=36) 
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1799)
    parser.add_argument('--save_total_limit', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--logging_steps', type=int, default=5)
    parser.add_argument('--save_steps', type=int, default=5)

    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--n_runs', type=int, default=1)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--upsample', default=False)

    # argument for the images
    parser.add_argument('--grid_layout', default=None)
    parser.add_argument('--image_size', default=None)
    parser.add_argument('--mask_patch_size', type=int, default=None)
    parser.add_argument('--mask_ratio', type=float, default=None)
    parser.add_argument('--mask_method', type=str, default=None)

    # argument for ablation study
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--finetune_mim', action='store_true')
    parser.add_argument('--freeze_vision_model', type=str, default="False")
    parser.add_argument('--freeze_text_model', type=str, default="False")
    parser.add_argument('--continue_training', action='store_true')

    # ----- NEW: Random baseline switches -----
    parser.add_argument('--run_random_baseline', action='store_true',
                        help='Also compute a random baseline on each split using current train class prior.')
    parser.add_argument('--random_kind', type=str, default='prior', choices=['prior', 'uniform'],
                        help='prior: Bernoulli with train prior; uniform: Bernoulli(0.5).')

    args = parser.parse_args()

    dataset = args.dataset
    dataset_prefix = args.dataset_prefix
    print(f'Dataset used: {dataset}, prefix: {dataset_prefix}.')
    
    upsample = args.upsample
    epochs = args.epochs
    image_size = grid_layout = None
    mask_patch_size = args.mask_patch_size
    mask_ratio = args.mask_ratio
    mask_method = args.mask_method
    freeze_vision_model = args.freeze_vision_model
    freeze_text_model = args.freeze_text_model
    if dataset == 'P12':
        base_path = '../../dataset/P12data'
        num_classes = 2
        upsample = True
        epochs = 4
        image_size = (384,384)
        grid_layout = (6,6)
    elif dataset == 'P19':
        base_path = '../../dataset/P19data'
        num_classes = 2
        upsample = True
        epochs = 2
        image_size = (384,384)
        grid_layout = (6,6)
    elif dataset == 'PAM':
        base_path = '../../dataset/PAMdata'
        num_classes = 8
        image_size = (256,320)
        grid_layout = (4,5)
        epochs = 20
    elif dataset == 'EthanolConcentration':
        base_path = '../../dataset/TSRAdata/Classification/EthanolConcentration'
        num_classes = 4
    elif dataset == 'Heartbeat':
        base_path = '../../dataset/TSRAdata/Classification/Heartbeat'
        num_classes = 2
    elif dataset == 'SpokenArabicDigits':
        base_path = '../../dataset/TSRAdata/Classification/SpokenArabicDigits'
        num_classes = 10
    elif dataset == 'SelfRegulationSCP1':
        base_path = '../../dataset/TSRAdata/Classification/SelfRegulationSCP1'
        num_classes = 2
    elif dataset == 'SelfRegulationSCP2':
        base_path = '../../dataset/TSRAdata/Classification/SelfRegulationSCP2'
        num_classes = 2
    elif dataset == 'Handwriting':
        base_path = '../../dataset/TSRAdata/Classification/Handwriting'
        num_classes = 26
    elif dataset == 'JapaneseVowels':
        base_path = '../../dataset/TSRAdata/Classification/JapaneseVowels'
        num_classes = 9
    elif dataset == 'UWaveGestureLibrary':
        base_path = '../../dataset/TSRAdata/Classification/UWaveGestureLibrary'
        num_classes = 8
    elif dataset == 'AppliancesEnergy':
        base_path = '../../dataset/TSRAdata/Regression/AppliancesEnergy'
        num_classes = 1
    elif dataset == 'BeijingPM10Quality':
        base_path = '../../dataset/TSRAdata/Regression/BeijingPM10Quality'
        num_classes = 1
    elif dataset == 'BeijingPM25Quality':
        base_path = '../../dataset/TSRAdata/Regression/BeijingPM25Quality'
        num_classes = 1
    elif dataset == 'LiveFuelMoistureContent':
        base_path = '../../dataset/TSRAdata/Regression/LiveFuelMoistureContent'
        num_classes = 1
    elif dataset == 'IEEEPPG':
        base_path = '../../dataset/TSRAdata/Regression/IEEEPPG'
        num_classes = 1
    elif dataset == 'BenzeneConcentration':
        base_path = '../../dataset/TSRAdata/Regression/BenzeneConcentration'
        num_classes = 1
    elif dataset == 'StudentLife':
        base_path = '../../dataset/studentlife_images'
        num_classes = 2  # binary classification
        upsample = False  # you can set this True if doing synthetic balance later
        epochs = 50
        image_size = (224, 224)  # matches ViT model expectation
        grid_layout = (6, 6)

    # prepare the model for vision-text classification
    image_model = args.image_model
    image_model_path = args.image_model_path
    if image_model == "vit": # default vit
        image_model_path = "google/vit-base-patch16-224-in21k"
        patch_size = 16
    elif image_model == "vit-384":
        image_model_path = "google/vit-base-patch16-384"
        patch_size = 16
    elif image_model == "swin": # default swin
        image_model_path = "microsoft/swin-base-patch4-window7-224-in22k"
        patch_size = 4
    elif image_model == "swin-224":
        image_model_path = "microsoft/swin-base-patch4-window7-224"
        patch_size = 4
    elif image_model == "resnet":
        image_model_path = "microsoft/resnet-50"

    text_model = args.text_model
    text_model_path = args.text_model_path
    if text_model == "longformer":
        text_model_path = "allenai/longformer-base-4096"
    elif text_model == "bigbird":
        text_model_path = "google/bigbird-roberta-base"
    if text_model == "clinical-longformer":
        text_model_path = "yikuan8/Clinical-Longformer"
    elif text_model == "clinical-bigbird":
        text_model_path = "yikuan8/Clinical-BigBird"
    elif text_model == "bert":
        text_model_path = "bert-base-uncased"
    elif text_model == "clinical-bert":
        text_model_path = "emilyalsentzer/Bio_ClinicalBERT"
    elif text_model == "roberta":
        text_model_path = "roberta-base"
    elif text_model == "bart":
        text_model_path = "facebook/bart-base"
    elif text_model == "electra":
        text_model_path = "google/electra-base-discriminator"
    max_length = 64

    feature_removal_level = args.feature_removal_level  # 'set' for fixed, 'sample' for random sample

    # While missing_ratio >0, feature_removal_level is automatically used
    if args.withmissingratio == True:
        missing_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
    else:
        missing_ratios = [0]
    print('missing ratio list', missing_ratios)

    for missing_ratio in missing_ratios:

        # prepare for training
        n_runs = args.n_runs
        n_splits = args.n_splits
        subset = False

        missing_ratio_arr = np.zeros((n_splits, n_runs))
        acc_arr = np.zeros((n_splits, n_runs))
        auprc_arr = np.zeros((n_splits, n_runs))
        auroc_arr = np.zeros((n_splits, n_runs))
        precision_arr = np.zeros((n_splits, n_runs))
        recall_arr = np.zeros((n_splits, n_runs))
        F1_arr = np.zeros((n_splits, n_runs))
        rmse_arr = np.zeros((n_splits, n_runs))
        mape_arr = np.zeros((n_splits, n_runs))
        mae_arr = np.zeros((n_splits, n_runs))

        # Arrays for random baseline (if enabled)
        rand_acc_arr = np.zeros((n_splits, n_runs))
        rand_auprc_arr = np.zeros((n_splits, n_runs))
        rand_auroc_arr = np.zeros((n_splits, n_runs))
        rand_precision_arr = np.zeros((n_splits, n_runs))
        rand_recall_arr = np.zeros((n_splits, n_runs))
        rand_F1_arr = np.zeros((n_splits, n_runs))

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
            elif dataset == 'PAM':
                split_path = '/splits/PAM_split_' + str(split_idx) + '.npy'
            elif dataset == 'StudentLife':
                split_path = os.path.join(base_path, 'splits', f'split_loo_{split_idx - 1}.npy')
            else:
                split_path = '/splits/split_' + str(split_idx) + '.npy'

            # find the pretrained mim image model 
            if args.finetune_mim:
                pretrained_image_model_dir = f"../../ckpt/ImgMIM/{dataset_prefix}{dataset}_{image_model}_{mask_patch_size}_{mask_ratio}_{mask_method}/split{split_idx}"
                if os.path.exists(pretrained_image_model_dir):
                    dir_list = os.listdir(pretrained_image_model_dir) # find the latest checkpoint
                    latest_checkpoint_idx = 0
                    for d in dir_list:
                        if "checkpoint" in d:
                            checkpoint_idx = int(d.split("-")[-1])
                            if checkpoint_idx > latest_checkpoint_idx:
                                latest_checkpoint_idx = checkpoint_idx

                    if latest_checkpoint_idx > 0 and os.path.exists(os.path.join(pretrained_image_model_dir, f"checkpoint-{latest_checkpoint_idx}")):
                        image_model_path = os.path.join(pretrained_image_model_dir, f"checkpoint-{latest_checkpoint_idx}")

            # the path to save models
            if args.output_dir is None:
                if args.finetune_mim:
                    output_dir = f"../../ckpt/VisionTextCLS/{dataset_prefix}{dataset}_{image_model}{freeze_vision_model}-{text_model}{freeze_text_model}_mim_{mask_patch_size}_{mask_ratio}_{mask_method}/split{split_idx}"
                else:
                    output_dir = f"../../ckpt/VisionTextCLS/{dataset_prefix}{dataset}_{image_model}{freeze_vision_model}-{text_model}{freeze_text_model}/split{split_idx}"
            else:
                output_dir = f"../../ckpt/VisionTextCLS/{args.output_dir}/split{split_idx}"

            # prepare the data:
            Ptrain, Pval, Ptest, ytrain, yval, ytest = get_data_split(
                base_path, split_path, split_idx, dataset=dataset, prefix=dataset_prefix, 
                upsample=upsample, missing_ratio=missing_ratio
            )
            print(len(Ptrain), len(Pval), len(Ptest), len(ytrain), len(yval), len(ytest))
            
            # if pval is none: use test dataset instead
            if len(Pval) == 0:
                Pval = Ptest
                yval = ytest
                print("Don't have val dataset, use test dataset as eval dataset instead")

            for m in range(n_runs):
                print('- - Run %d - -' % (m + 1))
                acc, precision, recall, F1, auc, aupr, rmse, mape, mae = fine_tune_hf(
                    image_model_path=image_model_path,
                    text_model_path=text_model_path,
                    freeze_vision_model=freeze_vision_model,
                    freeze_text_model=freeze_text_model,
                    output_dir=output_dir,
                    train_dataset=Ptrain,
                    val_dataset=Pval,
                    test_dataset=Ptest,
                    image_size=image_size,
                    grid_layout=grid_layout,
                    num_classes=num_classes,
                    max_length=max_length,
                    epochs=epochs,
                    train_batch_size=args.train_batch_size,
                    eval_batch_size=args.eval_batch_size,
                    logging_steps=args.logging_steps,
                    save_steps=args.save_steps,
                    learning_rate=args.learning_rate,
                    seed=args.seed,
                    save_total_limit=args.save_total_limit,
                    do_train=args.do_train,
                    continue_training=args.continue_training,
                    split_id=split_idx
                )
                
                test_report = 'Testing: Precision = %.2f | Recall = %.2f | F1 = %.2f\n' % (precision * 100, recall * 100, F1 * 100)
                test_report += 'Testing: AUROC = %.2f | AUPRC = %.2f | Accuracy = %.2f\n' % (auc * 100, aupr * 100, acc * 100)
                test_report += 'Testing: RMSE = %.2f | MAPE = %.2f | MAE = %.2f\n' % (rmse, mape, mae)
                print(test_report)
                
                result_path = "train_result.txt" if args.do_train else "test_result.txt"
                os.makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, result_path), "w+") as f:
                    f.write(test_report)

                # store testing results
                acc_arr[k, m] = acc * 100
                auprc_arr[k, m] = aupr * 100
                auroc_arr[k, m] = auc * 100
                precision_arr[k, m] = precision * 100
                recall_arr[k, m] = recall * 100
                F1_arr[k, m] = F1 * 100
                rmse_arr[k, m] = rmse
                mape_arr[k, m] = mape
                mae_arr[k, m] = mae

                # ---------- RANDOM BASELINE (optional) ----------
                if args.run_random_baseline and num_classes == 2:
                    # class prior from training labels (fallback to 0.5 if empty or nan)
                    try:
                        p1 = float(np.mean(np.asarray(ytrain).astype(int))) if len(ytrain) else 0.5
                        if np.isnan(p1): p1 = 0.5
                    except Exception:
                        p1 = 0.5
                    if args.random_kind == 'uniform':
                        p1 = 0.5

                    rng = np.random.RandomState(args.seed + split_idx * 1337 + m)
                    rand_scores = rng.rand(len(ytest))  # probabilistic scores in [0,1]
                    threshold = 1.0 - p1               # ensures P(ypred=1) ~= p1
                    ypred_rand = (rand_scores > threshold).astype(int)

                    r_acc, r_prec, r_rec, r_f1, r_auc, r_aupr = evaluate_binary_metrics(ytest, ypred_rand, rand_scores)

                    rand_report = (
                        f"[RandomBaseline | kind={args.random_kind} | p1={p1:.3f}] "
                        f"Prec={r_prec*100:.2f} | Rec={r_rec*100:.2f} | F1={r_f1*100:.2f} | "
                        f"AUROC={r_auc*100 if not np.isnan(r_auc) else float('nan'):.2f} | "
                        f"AUPRC={r_aupr*100 if not np.isnan(r_aupr) else float('nan'):.2f} | "
                        f"Acc={r_acc*100:.2f}"
                    )
                    print(rand_report)
                    with open(os.path.join(output_dir, "random_baseline_result.txt"), "w+") as f:
                        f.write(rand_report + "\n")

                    rand_acc_arr[k, m] = r_acc * 100
                    rand_auprc_arr[k, m] = (0.0 if np.isnan(r_aupr) else r_aupr * 100)
                    rand_auroc_arr[k, m] = (0.0 if np.isnan(r_auc) else r_auc * 100)
                    rand_precision_arr[k, m] = r_prec * 100
                    rand_recall_arr[k, m] = r_rec * 100
                    rand_F1_arr[k, m] = r_f1 * 100
                # -----------------------------------------------

        # pick best performer for each split based on max AUROC
        idx_max = np.argmax(auroc_arr, axis=1)
        acc_vec = [acc_arr[k, idx_max[k]] for k in range(n_splits)]
        auprc_vec = [auprc_arr[k, idx_max[k]] for k in range(n_splits)]
        auroc_vec = [auroc_arr[k, idx_max[k]] for k in range(n_splits)]
        precision_vec = [precision_arr[k, idx_max[k]] for k in range(n_splits)]
        recall_vec = [recall_arr[k, idx_max[k]] for k in range(n_splits)]
        F1_vec = [F1_arr[k, idx_max[k]] for k in range(n_splits)]
        rmse_vec = [rmse_arr[k, idx_max[k]] for k in range(n_splits)]
        mape_vec = [mape_arr[k, idx_max[k]] for k in range(n_splits)]
        mae_vec = [mae_arr[k, idx_max[k]] for k in range(n_splits)]

        mean_acc, std_acc = np.mean(acc_vec), np.std(acc_vec)
        mean_auprc, std_auprc = np.mean(auprc_vec), np.std(auprc_vec)
        mean_auroc, std_auroc = np.mean(auroc_vec), np.std(auroc_vec)
        mean_precision, std_precision = np.mean(precision_vec), np.std(precision_vec)
        mean_recall, std_recall = np.mean(recall_vec), np.std(recall_vec)
        mean_F1, std_F1 = np.mean(F1_vec), np.std(F1_vec)
        mean_rmse, std_rmse = np.mean(rmse_vec), np.std(rmse_vec)
        mean_mape, std_mape = np.mean(mape_vec), np.std(mape_vec)
        mean_mae, std_mae = np.mean(mae_vec), np.std(mae_vec)

        # printing the report
        test_report = "missing ratio:{}\n".format(missing_ratios)
        test_report += '------------------------------------------\n'
        test_report += 'Accuracy      = %.1f +/- %.1f\n' % (mean_acc, std_acc)
        test_report += 'AUPRC         = %.1f +/- %.1f\n' % (mean_auprc, std_auprc)
        test_report += 'AUROC         = %.1f +/- %.1f\n' % (mean_auroc, std_auroc)
        test_report += 'Precision     = %.1f +/- %.1f\n' % (mean_precision, std_precision)
        test_report += 'Recall        = %.1f +/- %.1f\n' % (mean_recall, std_recall)
        test_report += 'F1            = %.1f +/- %.1f\n' % (mean_F1, std_F1)
        test_report += 'RMSE          = %.1f +/- %.1f\n' % (mean_rmse, std_rmse)
        test_report += 'MAPE          = %.1f +/- %.1f\n' % (mean_mape, std_mape)
        test_report += 'MAE           = %.1f +/- %.1f\n' % (mean_mae, std_mae)

        # If random baseline ran, append its summary
        if args.run_random_baseline and num_classes == 2:
            r_idx_max = np.argmax(rand_auroc_arr, axis=1)
            r_acc_vec = [rand_acc_arr[k, r_idx_max[k]] for k in range(n_splits)]
            r_auprc_vec = [rand_auprc_arr[k, r_idx_max[k]] for k in range(n_splits)]
            r_auroc_vec = [rand_auroc_arr[k, r_idx_max[k]] for k in range(n_splits)]
            r_precision_vec = [rand_precision_arr[k, r_idx_max[k]] for k in range(n_splits)]
            r_recall_vec = [rand_recall_arr[k, r_idx_max[k]] for k in range(n_splits)]
            r_F1_vec = [rand_F1_arr[k, r_idx_max[k]] for k in range(n_splits)]

            test_report += '\n[Random Baseline Summary]\n'
            test_report += 'Accuracy      = %.1f +/- %.1f\n' % (np.mean(r_acc_vec), np.std(r_acc_vec))
            test_report += 'AUPRC         = %.1f +/- %.1f\n' % (np.mean(r_auprc_vec), np.std(r_auprc_vec))
            test_report += 'AUROC         = %.1f +/- %.1f\n' % (np.mean(r_auroc_vec), np.std(r_auroc_vec))
            test_report += 'Precision     = %.1f +/- %.1f\n' % (np.mean(r_precision_vec), np.std(r_precision_vec))
            test_report += 'Recall        = %.1f +/- %.1f\n' % (np.mean(r_recall_vec), np.std(r_recall_vec))
            test_report += 'F1            = %.1f +/- %.1f\n' % (np.mean(r_F1_vec), np.std(r_F1_vec))

        print(test_report)

        os.makedirs(output_dir.split("split")[0], exist_ok=True)
        with open(os.path.join(output_dir.split("split")[0], "test_result.txt"), "w+") as f:
            f.write(test_report)
