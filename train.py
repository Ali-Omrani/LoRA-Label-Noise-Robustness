from __future__ import absolute_import, division, print_function
import IPython
import wandb
import argparse
import os
import numpy as np
import pandas as pd
from sklearn import metrics
from tqdm import tqdm
import transformers
import torch
import json
from torch import cuda
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
from transformers import BertTokenizer, BertModel, BertConfig
from sklearn.model_selection import train_test_split
from custom_dataset import CustomDataset
from utils import (
    compute_metrics, 
    create_logger, 
    set_seed, 
    get_dataset_loader_func, 
    EvalOnTrainCallback, 
    LogPredicitonsCallback,
    label2dataset, 
    dataset2label,
    introduce_noise,
    get_trainer,
    add_label_noise)

from transformers import ProgressCallback
from datasets import Dataset as HFDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    AdamW,
    get_linear_schedule_with_warmup
)
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PromptEncoderConfig,
    PromptTuningConfig,
    LoraConfig,
)
from datasets import load_dataset, interleave_datasets

# from consts import get_data_dirs_cardinal, get_processors, get_memories, get_reporters

# from consts import get_base_parameters_trainer, get_models, get_configs, get_tokenizers

import logging
import random

import numpy as np
import torch
from tqdm import tqdm, trange

from datetime import datetime

from custom_trainer import CustomTrainer





def get_datasets(data, label_col, tokenize_function, train_size = 0.8, sampling_strategy="oversampling", oversampling_probs=None, logger=None, noise_ratio=None, noisy_eval=False):


    if type(data) is dict:
        logger.info("Using pre-defined train/val/test split")
        train_df = data["train"]
        val_df = data["val"]
        test_df = data["test"]
    
    
    else:
        logger.info(f"Splitting the data into train/val/test with stratification on {label_col}")
    # Split the DataFrame into a training and a test set while maintaining the label proportions.
        train_df, rest_df = train_test_split(data, test_size=1-train_size, stratify=data[label_col], random_state=42)
        val_df, test_df = train_test_split(rest_df, test_size=0.5, stratify=rest_df[label_col], random_state=42)
   
    train_df = train_df[["text",   label_col]]
    val_df = val_df[["text",   label_col]]
    test_df = test_df[["text",   label_col]]

    logger.info(f"Train size: {len(train_df)}")
    logger.info(f"support of {label_col} in train: {sum(train_df[label_col])}")
    logger.info(f"Val size: {len(val_df)}")
    logger.info(f"support of {label_col} in val: {sum(val_df[label_col])}")
    logger.info(f"Test size: {len(test_df)}")
    logger.info(f"support of {label_col} in test: {sum(test_df[label_col])}")
    
    pos_freq = sum(train_df[label_col]) 
    # setting lable weights to be inverse of the frequency of the label in the training set
    label_weights = [len(train_df)/(len(train_df)-pos_freq), len(train_df)/(pos_freq)]



        
    
            
    if "sampling" in sampling_strategy:
        label_weights= [1.0,1.0] # overriding the label weights to be 1,1 for oversampling
        logger.info(f"setting label weights to {label_weights} because we're using a sampling strategy: {sampling_strategy}")
        oversampling_probs = [0.5, 0.5] if oversampling_probs is None else oversampling_probs
        if sampling_strategy == "oversampling":
            stopping_strategy = "all_exhausted"
            logger.info(f"Oversampling with probabilities {oversampling_probs} and stopping strategy with {stopping_strategy}")
        elif sampling_strategy == "undersampling":
            stopping_strategy = "first_exhausted"
            logger.info(f"Undersampling with probabilities {oversampling_probs} and stopping strategy with {stopping_strategy}")
        train_dataset_pos = HFDataset.from_pandas(train_df[train_df[label_col] == 1])
        train_dataset_neg = HFDataset.from_pandas(train_df[train_df[label_col] == 0])
        # adding noise
        if noise_ratio:
            logger.info("adding noise before sampling")
            train_dataset_pos =train_dataset_pos.map(lambda example: add_label_noise(example, noise_ratio, label_col))
            train_dataset_neg =train_dataset_neg.map(lambda example: add_label_noise(example, noise_ratio, label_col))

        train_dataset = interleave_datasets([train_dataset_pos, train_dataset_neg], probabilities=oversampling_probs, seed=42, stopping_strategy=stopping_strategy)
        logger.info(f"Train size after sampling: {len(train_dataset)} with N-positive = {sum(train_dataset[label_col])}")
    else:
        train_dataset = HFDataset.from_pandas(train_df)
        if noise_ratio:
            train_dataset =train_dataset.map(lambda example: add_label_noise(example, noise_ratio, label_col))
    
    val_dataset = HFDataset.from_pandas(val_df)
    if noisy_eval:
        val_dataset =val_dataset.map(lambda example: add_label_noise(example, noise_ratio, label_col))

    
    
    test_dataset = HFDataset.from_pandas(test_df)

    

    train_tokenized_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    # remove_columns=["text_id"],
    )

    val_tokenized_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        # remove_columns=["text_id"],
    )

    test_tokenized_dataset = test_dataset.map(
        tokenize_function,
        batched=True,
        # remove_columns=["text_id"],
    )

    train_tokenized_dataset = train_tokenized_dataset.rename_column(label_col, "label")
    val_tokenized_dataset = val_tokenized_dataset.rename_column(label_col, "label")
    test_tokenized_dataset = test_tokenized_dataset.rename_column(label_col, "label")
    

    return train_tokenized_dataset, val_tokenized_dataset, test_tokenized_dataset, label_weights

def setup_tokenizer(model_name_or_path):
    if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
        padding_side = "left"
    else:
        padding_side = "right"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id


    def tokenize_function(examples):
        # max_length=None => use the model's max length (it's actually the default)
        outputs = tokenizer(examples["text"], truncation=True, max_length=400)
        return outputs
    
    
    return tokenizer, tokenize_function


def balance_df(df, label_col, balance_ratio=0.5):
    pos_df = df[df[label_col]==1]
    return pd.concat([pos_df, df[df[label_col]==0].sample(int(((1/balance_ratio)-1)*len(pos_df)))])

def downsample_balance(df, label_col, balance_ratio=0.5, limited_data=None):

    # Separate rows where label_col is 0 and 1
    df_label_0 = df[df[label_col] == 0]
    df_label_1 = df[df[label_col] == 1]
    if limited_data:
        num_rows_label_1 = int(balance_ratio*limited_data)
        num_rows_label_0 = limited_data - num_rows_label_1
        df_balanced = pd.concat([df_label_0.sample(n=num_rows_label_0), df_label_1.sample(n=num_rows_label_1)])
    else:
    # Calculate the number of rows to keep for label_col = 1 based on balance_ratio
        num_rows_label_1 = int(len(df_label_0) * balance_ratio)
    
    # Sample a balanced subset for label_col = 1
        df_balanced = pd.concat([df_label_0, df_label_1.sample(n=num_rows_label_1, replace=True)])
    
    return df_balanced

def log_to_wandb(log_dict, wandb_stop=False):
    print("in here wandb_stop", wandb_stop)
    if not wandb_stop:
        print("logging test shit")
        wandb.log(log_dict)

def train(args):
   
    # ----- create loggers and wandb run -----
    run_name= f"{args.label_col}-balance-{args.balance_ratio}"
    if not args.wandb_stop:
        wandb.init(project=args.project_name, name=run_name, config=args)
   
    output_directory = os.path.join("experiments", args.experiment_subdir, f"{args.method}-{args.label_col}-noise-{args.noise_ratio}")
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    args_json = json.dumps(vars(args), indent=4)
    args_file_path = os.path.join(output_directory, "args.json")
    with open(args_file_path, "w") as args_file:
        args_file.write(args_json)
    
    logger = create_logger(output_directory)
    logger.info(args)
    
    device = torch.device("cuda" if cuda.is_available() else "cpu")

    logger.info(f"Using {device} device")
    
    # -------------- load data, balance (if needed), add noise (if needed) 
    all_annotations_df = get_dataset_loader_func(label2dataset[args.label_col])
    log_to_wandb({"dataset_name": label2dataset[args.label_col]}, wandb_stop=args.wandb_stop)
    
    if type(all_annotations_df) is dict :
        if args.limited_data and len(all_annotations_df["train"]) > args.limited_data:
            if args.balance:
            # all_annotations_df["train"] = balance_df(all_annotations_df["train"], args.label_col, balance_ratio=args.balance_ratio)
                all_annotations_df["train"] = downsample_balance(all_annotations_df["train"], args.label_col, balance_ratio=args.balance_ratio, limited_data=args.limited_data)
            else:
                all_annotations_df["train"] = all_annotations_df["train"].sample(args.limited_data)
        if args.DEV:
            for split in ["train", "val", "test"]:
                all_annotations_df[split] = all_annotations_df[split].sample(300)

    else:
        if args.balance:
            all_annotations_df = balance_df(all_annotations_df, args.label_col, balance_ratio=args.balance_ratio)

        if  args.limited_data and len(all_annotations_df) > args.limited_data:
            all_annotations_df = all_annotations_df.sample(args.limited_data)



    # get majority and drop non-essential coloumns
    if args.get_majority:
        df = all_annotations_df[["text",  "annotator", args.label_col]]

        majority_df = df.groupby('text_id').agg({
                'text': 'first',  # Keep the first 'text' value
                args.label_col: lambda x: x.mode().iloc[0],  # Calculate mode for the 'label' column
            }).reset_index()
        exists_df = df.groupby('text_id').agg({
                'text': 'first',  # Keep the first 'text' value
                args.label_col:     lambda x: 1 if any(x == 1) else 0,  # Calculate mode for the 'label' column
            }).reset_index()
        
        df = majority_df
    else:
        df = all_annotations_df

    # if args.DEV:
    #         df = df.sample(2000)

    # args_dict = {attr: getattr(args, attr) for attr in dir(args) if not callable(getattr(args, attr)) and not attr.startswith("__")}

    # if args.noise_ratio:
    #     all_annotations_df["train"] = introduce_noise(all_annotations_df["train"], args.label_col, noise_ratio=args.noise_ratio)
    # if args.noisy_eval:
    #     all_annotations_df["val"] = introduce_noise(all_annotations_df["val"], args.label_col, noise_ratio=args.noise_ratio)
           
    # wandb.log(args_dict)
    tokenizer, tokenize_function = setup_tokenizer(args.LM)
    train_dataset, val_dataset, test_dataset, label_weights = \
        get_datasets(df, args.label_col, tokenize_function, train_size=0.8, sampling_strategy=args.imbalance_strategy, \
                      oversampling_probs=[args.sampling_ratio, 1-args.sampling_ratio], logger=logger, noise_ratio=args.noise_ratio, noisy_eval=args.noisy_eval)
    

    if args.imbalance_strategy != "weightedloss":
        label_weights = [1.0, 1.0]
    
    logger.info(f"Label weights: {label_weights}, imbalance strategy: {args.imbalance_strategy}")

    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    
    # -------------- Set up Trainer
    
    peft_config = None
    if args.method =="p_tuning":
        peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=20, encoder_hidden_size=128)
    elif args.method == "lora":
        peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=args.LORA_r, lora_alpha=args.LORA_alpha, lora_dropout=0.1)
    elif args.method == "prompt_tuning":
        peft_config = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)



    model = AutoModelForSequenceClassification.from_pretrained(args.LM, return_dict=True)
    if peft_config:
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    logger.info(f"Using {args.method} method")
    
    

    training_args = TrainingArguments(
        output_dir=output_directory,
        learning_rate=args.LEARNING_RATE,
        per_device_train_batch_size=args.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=args.VALID_BATCH_SIZE,
        num_train_epochs=args.EPOCHS,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        logging_first_step=True,
        load_best_model_at_end=True,
        # lr_scheduler_type="linear" if  args.method =="lora" else None,
        warmup_ratio=args.warmup_ratio,
        # metric_for_best_model="auc_roc",
        report_to="wandb" if not args.wandb_stop else None,  # enable logging to W&B
        run_name=run_name,
        save_total_limit=1,
    )
    
    # # Define the optimizer
    # optimizer = AdamW(
    #     model.parameters(),
    #     lr=training_args.learning_rate,
    #     weight_decay=training_args.weight_decay,
    # )

    # # Add linear warmup scheduler
    # num_warmup_steps = int(0.1 * args.EPOCHS * len(train_dataset) / training_args.per_device_train_batch_size)
    # num_training_steps = int(args.EPOCHS * len(train_dataset) / training_args.per_device_train_batch_size)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    if args.imbalance_strategy == "focalloss":
        args.trainer_class = "withfocalloss"
    trainer_class = get_trainer(args.trainer_class)
    logger.info(f"Using trainer cllass {trainer_class}")
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        noise_ratio=args.noise_ratio,
        # callbacks=[ProgressCallback],
        label_weights=torch.tensor(label_weights).to(device),
        # optimizers=(optimizer, scheduler),
    )
    trainer.add_callback(EvalOnTrainCallback(trainer)) 
    trainer.add_callback(LogPredicitonsCallback(logger, trainer, output_directory))


    trainer.train()
    trainer.save_model(os.path.join(output_directory, "checkpoint-best"))


    # -------------- Test 

    res = trainer.predict(test_dataset)
    test_metrics = res.metrics
    test_metrics = {"test/"+k[5:]: v for k, v in test_metrics.items()}
    print("logging test metrics to wandb", test_metrics)
    log_to_wandb(test_metrics, wandb_stop=args.wandb_stop)


def sweep_train(config=None):
    
    with wandb.init(config=config):

        args=wandb.config
        train(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DEV", action="store_true", help="Whether it's a development run")
    parser.add_argument("--get_majority", action="store_true", help="Whether to compute a majority vote dataframe or not")
    parser.add_argument("--train_size", type=float, default=0.8, help="Training data split size")
    parser.add_argument("--sampling_ratio", type=float, default=0.3, help="Upsampling ratio for minority class")
    parser.add_argument("--MAX_LEN", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--TRAIN_BATCH_SIZE", type=int, default=32, help="Training batch size")
    parser.add_argument("--VALID_BATCH_SIZE", type=int, default=64, help="Validation batch size")
    parser.add_argument("--LEARNING_RATE", type=float, default=1e-05, help="Learning rate")
    parser.add_argument("--label_col", type=str, default="VO", help="Label column name")
    parser.add_argument("--EPOCHS", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--LM", type=str, default="roberta-base", help="the pretrained language model to use")
    parser.add_argument("--method", type=str, default="finetune", help="the method to use for training")
    parser.add_argument("--dataset_name", type=str, default="personal_attack", help="the dataset for training")
    parser.add_argument("--imbalance_strategy", type=str, default="nothing", help="the strategy to mitigate label imbalance. Options oversampling, undersampling, weightedloss, focalloss")
    parser.add_argument("--project_name", type=str, default="noise-naacl", help="project name for wandb")
    parser.add_argument("--sweep", action="store_true", help="Whether to do sweep or not")
    parser.add_argument("--sweep_name", type=str, default=None, help="the name for sweep (only with --sweep)")
    parser.add_argument("--balance", action="store_true", help="create a balanced dataset ")
    parser.add_argument("--balance_ratio", type=float, default=0.5, help="ratio of positive class in balanced experiments (only works with --balance)")
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="ratio for scheduler warmup")
    parser.add_argument("--warmup", action="store_true", help="use linear schduler with warmup")
    parser.add_argument("--limited_data", type=int, default=None, help="whether to use limited data and how much should the limit be")
    parser.add_argument("--LORA_r", type=int, default=8, help="lora R")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="l2 regularizatoin - used as weight decay param in AdamW")
    parser.add_argument("--LORA_alpha", type=int, default=16, help="lora R")
    parser.add_argument("--noise_ratio", type=float, default=None, help="ratio of training samples that will get their labels flipped")
    parser.add_argument("--trainer_class", type=str, default=None, help="the method to use for training")
    parser.add_argument("--noisy_eval", action="store_true", help="Whether to add noise to the evaluation set")
    parser.add_argument("--experiment_subdir", type=str, default=None, help="title of subdir for experiment")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducability")
    parser.add_argument("--wandb_stop", action="store_true", help="Whether to use wandb or not")


    
    args = parser.parse_args()
    

    # subdir = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.experiment_subdir}"
    subdir = args.experiment_subdir
    experiment_path = os.path.join("experiments", subdir)

    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)
    args.experiment_subdir = subdir

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.device_count() > 0:
            torch.cuda.manual_seed_all(args.seed)
    print(dataset2label[args.dataset_name])
    run_name= f"{args.label_col}-balance-{args.balance_ratio}"

    if not args.sweep:
        print("NO SWEEP!!!!!!!!!!!!!!!!!!!!!!!!")
        print(args)
        # ------------- Make Train/Val/Test Dataloaders
        train(args)
    else:

    # setting up sweep

        hyperparameter_defaults = vars(args)

        sweep_hyperparameter_defaults = {k: {"value":hyperparameter_defaults[k]} for k in hyperparameter_defaults}
        
        # ---------------- sweep -----------------
        sweep_config = {
        'method': 'grid'
        }

        metric = {
        'name': 'eval/loss',
        'goal': 'minimize'   
        }

        sweep_config['metric'] = metric

        parameters_dict = sweep_hyperparameter_defaults

        parameters_dict.update({
        
        # 'imbalance_strategy': {
        #    'values': ["oversampling","undersampling", "weightedloss"]
        # },
        # 'EPOCHS': {
        #     'values': [10, 20, 30, 40]
        # },
        # 'warmup_ratio': {
        #     'values': [0.06]
        # },
         'noise_ratio': {
            'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        },
        # 'balance_ratio': {
        #     'values': [0.1, 0.2, 0.3, 0.4, 0.5]
        # },
        # 'method':{
        #     'values': ["finetune"]
        # },
        # 'LEARNING_RATE' :{
        #     # 'values': [1e-4, 5e-4, 1e-3, 1e-5, 5e-5]
        #     'values': [1e-5]
        # },
        # 'TRAIN_BATCH_SIZE':{
        #  'values': [16, 32, 64, 128]
        # },
        # 'LORA_alpha':{
        #     'values': [8, 16]
        # },
        'label_col':{
            # 'values': ["vo", "hd", "cv"]
            # 'values': ["antagonize", "condescending", "hostile"]
            # 'values':[  'tpa', 'oa', 'ra', "antagonize", "condescending", "hostile", "vo", "hd", "cv"]
            'values': dataset2label[args.dataset_name]
            
        }})
        
        
        if args.sweep_name is not None:
            sweep_config.update({"name": args.sweep_name})
        print(sweep_config)
        sweep_config['parameters'] = parameters_dict
        sweep_id = wandb.sweep(sweep_config, project=args.project_name)

        wandb.agent(sweep_id, sweep_train, count=45)



