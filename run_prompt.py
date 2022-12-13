# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import logging
import math
import os
import random
import shutil
import time
import json

import datasets
from datasets import load_dataset
from evaluate import load
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)

from models import RobertaForPromptFinetuning, AutoModelForPromptFinetuning, BertForPromptFinetuning, DebertaV2ForPromptFinetuning, OPTForPromptFinetuning

from utils import task_input_key, task_label_key, task_metric
from prompting import MultiLabelPrompting

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default=None, 
        required=True, 
        help="A dictionary containing the training, validation, test data."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.0, 
        help="Weight decay to use.")
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=3, 
        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", 
        type=int, 
        default=0, 
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None, 
        help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--shot_num", 
        type=int, 
        default=None, 
        required=True,
        help="The number of shots to use for training."    
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=10, 
        help="Select top k label token for each class."
    )
    parser.add_argument(
        "--eval_steps", 
        type=int, 
        default=None,
        help="The number of steps to use for evaluation."
    )
    parser.add_argument(
        "--logging_loss_steps", 
        type=int, 
        default=10,
        help="The number of steps to use for logging the loss."
    )
    parser.add_argument(
        "--template", 
        type=str, 
        default=None,
        help="The template to use for the output file."
    )
    parser.add_argument(
        "--dedup", 
        action="store_true",
        default=False,
        help="Whether to dedup label tokens."
    )
    parser.add_argument(
        "--random_k_token", 
        action='store_true', 
        default=False,
        help="Whether to random select k label tokens."
    )
    parser.add_argument(
        "label_token_mode",
        type=str,
        choices=["AMuLaP", "AutoL", "PETAL"],
        default="AMuLaP",
        help="How to get the label token."
    )
    parser.add_argument(
        "--mapping_path",
        type=str,
        default=None,
        help="The path to the label token mapping file."
    )
    parser.add_argument(
        "--max_seq_len", 
        type=int, 
        default=128,
        help="The maximum sequence length."
    )
    parser.add_argument(
        "--first_sent_limit", 
        type=int, 
        default=None,
        help="The maximum first sentence length."
    )
    parser.add_argument(
        "--other_sent_limit", 
        type=int, 
        default=None,
        help="The maximum other sentence length."
    )
    parser.add_argument(
        "--no_finetune",
        action="store_true",
        default=False,
        help="Whether to finetune the model."
    )
    parser.add_argument(
        "--exp_id",
        type=str,
        default=0,
        help="Experiment ID"
    )
    parser.add_argument(
        "--temp_dir",
        type=str,
        default=None,
        help="Experiment ID"
    )
    args = parser.parse_args()

    if args.output_dir is not None:
        args.logging_dir = os.path.join(args.output_dir, "logging", args.task_name, str(args.shot_num) + "-" + str(args.seed))
        os.makedirs(args.logging_dir, exist_ok=True)
        if not args.no_finetune:
            dir_name = "trainstep{}_warmupstep{}_lr{}_pbs{}".format(args.max_train_steps, args.num_warmup_steps, args.learning_rate, args.per_device_train_batch_size)
            dir_name += "_topk{}".format(args.top_k)
            dir_name += "_" + args.label_token_mode
            if args.label_token_mode == "AMuLaP":
                dir_name += "_random" if args.random_k_token else ""
                dir_name += "_dedup" if args.dedup else ""
            args.output_dir = os.path.join(args.output_dir, args.task_name, str(args.shot_num) + "-" + str(args.seed), dir_name)
            os.makedirs(args.output_dir, exist_ok=True)
    
    return args

def load_data(task_name, data_dir):
    if task_name in ["sst2", "cola", "mrpc", "qnli", "qqp", "rte", "trump", "debat"]:
        data_files = {
            "train": os.path.join(data_dir, "train.tsv"),
            "dev": os.path.join(data_dir, "dev.tsv"),
            "test": os.path.join(data_dir, "test.tsv"),
        }
    elif task_name in ["mnli"]:
        data_files = {
            "train": os.path.join(data_dir, "train.tsv"),
            "dev": os.path.join(data_dir, "dev_matched.tsv"),
            "test_m": os.path.join(data_dir, "test_matched.tsv"),
            "test_mm": os.path.join(data_dir, "test_mismatched.tsv"),
        }
        
    if task_name in ["sst2", "mnli", "mrpc", "qnli", "qqp", "rte", "trump", "debat"]:
        dataset = load_dataset('csv', data_files=data_files, delimiter='\t', quoting=3)
    elif task_name in ["cola"]:
        dataset = load_dataset('csv', data_files=data_files, delimiter='\t', column_names=["id", "label", "_", "sentence"])

    return dataset

def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


def main():
    args = parse_args()
    
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    # logging_dir = os.path.join(args.output_dir, "logging", args.task_name, str(args.shot_num) + "-" + str(args.seed))
    # os.makedirs(logging_dir, exist_ok=True)
    filename = None
    if args.no_finetune:
        filename = "no_finetune"
    else:
        filename = "trainstep{}_warmupstep{}_lr{}_pbs{}".format(args.max_train_steps, args.num_warmup_steps, args.learning_rate, args.per_device_train_batch_size)
    filename += "_topk{}".format(args.top_k)
    filename += "_" + args.label_token_mode
    if args.label_token_mode == "AMuLaP":
        filename += "_random" if args.random_k_token else ""
        filename += "_dedup" if args.dedup else ""
    filename += ".log"
    logging_filename = os.path.join(args.logging_dir, filename)
    logging.basicConfig(
        filename=logging_filename,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        torch.manual_seed(args.seed)

    raw_datasets = load_data(args.task_name, args.data_dir)

    label2id = None
    labels = raw_datasets["train"][task_label_key[args.task_name]]
    labels = list(set(labels))
    label2id = {label: i for i, label in enumerate(labels)}
    
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=len(label2id), finetuning_task=args.task_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    
    if "roberta" in args.model_name_or_path:
        model = RobertaForPromptFinetuning.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    elif "bert-" in args.model_name_or_path:
        model = BertForPromptFinetuning.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    elif "opt" in args.model_name_or_path:
        model = OPForPromptFinetuning.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    elif "deberta-v2" in args.model_name_or_path:
        model = DebertaV2ForPromptFinetuning.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        model = AutoModelForPromptFinetuning.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )

    multi_label_prompting = MultiLabelPrompting(model, tokenizer, label2id)

    processed_datasets = raw_datasets.map(
        lambda example : multi_label_prompting.preprocess(
            example,
            input_key = task_input_key[args.task_name], 
            label_key = task_label_key[args.task_name], 
            max_length = args.max_seq_len,
            template = args.template,
            first_sent_limit=args.first_sent_limit,
            other_sent_limit=args.other_sent_limit,
        ),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )


    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["dev"]
    

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    data_collator = default_data_collator

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    multi_label_prompting.model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    k_map = multi_label_prompting.top_k_indices(train_dataloader, eval_dataloader, args.top_k, args.shot_num, args.label_token_mode, args.mapping_path, args.dedup, args.random_k_token)
    #multi_label_prompting.manual_labeling({0: ["disappointing", "awful", "terrible", "boring", "horrible"], 1: ["amazing", "great", "brilliant", "wonderful", "fantastic"]})
    model = multi_label_prompting.model
    tokenizer = multi_label_prompting.tokenizer

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Get the metric function
    task_name_ = args.task_name
    if (task_name_ == "trump" or task_name_ == "debat"):
        task_name_ = "sst2"
    metric = load(task_metric[args.task_name], experiment_id = args.exp_id)

    if args.no_finetune:
        for split in ["dev", "test"] if args.task_name != "mnli" else ["dev", "test_m", "test_mm"]:
            eval_dataset = processed_datasets[split]
            eval_dataloader = DataLoader(
                eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
            )
            eval_dataloader = accelerator.prepare(eval_dataloader)

            model.eval()
            for step, batch in enumerate(eval_dataloader):
                outputs = model(batch["input_ids"], batch["attention_mask"], batch["mask_pos"])
                predictions = outputs[0].argmax(dim=-1)
                metric.add_batch(
                    predictions=accelerator.gather(predictions),
                    references=accelerator.gather(batch["labels"]),
                )

            eval_metric = metric.compute()
            logger.info(f"{split}: {eval_metric}")
        return

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    best_metric = -1
    best_metric_step = None
    
    if args.temp_dir is not None:
        pth = args.temp_dir + '/' + args.output_dir
    else:
        pth = args.output_dir
    
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(batch["input_ids"], batch["attention_mask"], batch["mask_pos"], batch["labels"])
            loss = outputs[0]
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps % args.logging_loss_steps == 0 and completed_steps != 0:
                logger.info(f"step {completed_steps}: loss: {loss}")

            if completed_steps >= args.max_train_steps:
                break

            if args.eval_steps is not None:
                if completed_steps % args.eval_steps == 0 and completed_steps != 0:
                    model.eval()
                    for eval_step, eval_batch in enumerate(eval_dataloader):
                        outputs = model(eval_batch["input_ids"], eval_batch["attention_mask"], eval_batch["mask_pos"])
                        predictions = outputs[0].argmax(dim=-1)
                        metric.add_batch(
                            predictions=accelerator.gather(predictions),
                            references=accelerator.gather(eval_batch["labels"]),
                        )

                    eval_metric = metric.compute()
                    logger.info(f"step {completed_steps}: {eval_metric}")
                    if eval_metric[task_metric[args.task_name]] >= best_metric:
                        best_metric = eval_metric[task_metric[args.task_name]]
                        best_metric_step = completed_steps
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(pth, save_function=accelerator.save)
                    model.train()

    model.eval()
    for eval_step, eval_batch in enumerate(eval_dataloader):
        outputs = model(eval_batch["input_ids"], eval_batch["attention_mask"], eval_batch["mask_pos"])
        predictions = outputs[0].argmax(dim=-1)
        metric.add_batch(
            predictions=accelerator.gather(predictions),
            references=accelerator.gather(eval_batch["labels"]),
        )

    eval_metric = metric.compute()
    logger.info(f"step {completed_steps}: {eval_metric}")
    if eval_metric[task_metric[args.task_name]] > best_metric:
        best_metric = eval_metric[task_metric[args.task_name]]
        best_metric_step = completed_steps
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(pth, save_function=accelerator.save)

    logger.info(f"early stop at step {best_metric_step}, metric: {best_metric}")
    
    if "roberta" in args.model_name_or_path:
        model = RobertaForPromptFinetuning.from_pretrained(
            pth,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    elif "bert-" in args.model_name_or_path:
        model = BertForPromptFinetuning.from_pretrained(
            pth,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    elif "opt" in args.model_name_or_path:
        model = OPForPromptFinetuning.from_pretrained(
            pth,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    elif "deberta-v2" in args.model_name_or_path:
        model = DebertaV2ForPromptFinetuning.from_pretrained(
            pth,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        model = AutoModelForPromptFinetuning.from_pretrained(
            pth,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )

    model.label_token_list = {}
    for key in label2id:
        label = label2id[key]
        model.label_token_list[label] = torch.tensor(k_map[label]).long().cuda()
    model = accelerator.prepare(model)

    for split in ["test"] if args.task_name != "mnli" else ["test_m", "test_mm"]:
        eval_dataset = processed_datasets[split]
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )
        eval_dataloader = accelerator.prepare(eval_dataloader)

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            outputs = model(batch["input_ids"], batch["attention_mask"], batch["mask_pos"])
            predictions = outputs[0].argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

        eval_metric = metric.compute()
        logger.info(f"{split}: {eval_metric}")
    

    # Delete the saved outputs to save space
    shutil.rmtree(args.output_dir)

if __name__ == "__main__":
    main()