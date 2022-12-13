import logging
import math
import os
import random
import shutil
import time
import json

import datasets
from datasets import load_dataset, load_metric
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)


class MultiLabelPrompting(object):
    def __init__(
        self, 
        model, 
        tokenizer, 
        label2id,
    ):
        self.model = model
        self.tokenizer = tokenizer
        
        self.label2id = label2id

    
    def create_optimizer(self, optimizer = AdamW, lr = 0.00001, weight_decay = 0.001):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        return optimizer(optimizer_grouped_parameters, lr=lr)
    
    def manual_labeling(self, labels):
        self.model.label_token_list = {}
        for key in self.label2id:
            label = self.label2id[key]
            
            tokens = []
            for l in labels[label]:
                tokens.append(self.tokenizer(l).input_ids[1])
            self.model.label_token_list[label] = torch.tensor(tokens).long().cuda()
    
    def top_k_indices(self, train_dataloader, eval_dataloader, top_k, shot_num, label_mode = "AMuLaP", mapping_path = None, dedup = False, random_k_token = False):        
        # get top-k token index
        k_map = {}
        if label_mode == "AMuLaP":
            self.model.eval()
            all_train_logits = {}
            with torch.no_grad():
                for step, batch in enumerate(train_dataloader):
                    outputs = self.model(batch["input_ids"], batch["attention_mask"], batch["mask_pos"])
                    for i in range(len(batch["labels"])):
                        if batch["labels"][i].item() not in all_train_logits:
                            all_train_logits[batch["labels"][i].item()] = outputs[0][i].cpu()
                        else:
                            all_train_logits[batch["labels"][i].item()] += outputs[0][i].cpu()

            map_index = {}
            for key in self.label2id:
                label = self.label2id[key]
                all_train_logits[label] = all_train_logits[label] / shot_num
                sorted_logits, sort_index = torch.sort(all_train_logits[label], descending=True)
                map_index[label] = sort_index.tolist()
            
            label_token_set = {}
            for key in self.label2id:
                label_token_set[self.label2id[key]] = []

            for i in range(self.tokenizer.vocab_size):
                logits = [all_train_logits[self.label2id[key]][i] for key in self.label2id]
                label_token_set[logits.index(max(logits))].append({
                    "idx": i,
                    "prob": max(logits),
                })

            def myFunc(e):
                return e['prob']
        
            for key in self.label2id:
                label = self.label2id[key]
                label_token_set[label].sort(reverse=True, key=myFunc)

            if dedup:
                for key in self.label2id:
                    label = self.label2id[key]
                    k_map[label] = []
                    for i in range(top_k):
                        k_map[label].append(label_token_set[label][i]["idx"])
            elif random_k_token:
                num_list = random.sample(range(self.tokenizer.vocab_size), top_k * len(self.label2id))
                for i, key in enumerate(self.label2id):
                    label = self.label2id[key]
                    k_map[label] = num_list[i * top_k : (i+1) * top_k]
            else:
                for key in self.label2id:
                    label = self.label2id[key]
                    k_map[label] = map_index[label][:top_k]
        elif label_mode == "AutoL":
            label_to_word = {}
            for key in self.label2id:
                label = self.label2id[key]
                label_to_word[label] = []
            # seed_mapping_path = os.path.join(mapping_path, "{}-{}.sort.txt".format(shot_num, args.seed))
            with open(mapping_path) as f:
                for line in f:
                    line = line.strip()
                    line = eval(line)
                    for key in line:
                        word = line[key]
                        if word[0] not in ['<', '[', '.', ',']:
                            assert len(self.tokenizer.tokenize(' ' + word)) == 1
                            word = self.tokenizer._convert_token_to_id(self.tokenizer.tokenize(' ' + word)[0])
                        else:
                            word = self.tokenizer._convert_token_to_id(word)
                        if len(key) == 1:
                            label_to_word[self.label2id[int(key)]].append(word)
                        else:
                            label_to_word[self.label2id[key]].append(word)
            for key in self.label2id:
                label = self.label2id[key]
                k_map[label] = label_to_word[label][:top_k]
        elif label_mode == "PETAL":
            label_to_word = {}
            for key in self.label2id:
                label = self.label2id[key]
                label_to_word[label] = []
            # seed_mapping_path = os.path.join(mapping_path, "{}-{}.json".format(shot_num, args.seed))
            with open(mapping_path) as f:
                for line in f:
                    line = line.strip()
                    line = json.loads(line)
                    for key in line:
                        for word in line[key]:
                            if word[0] not in ['<', '[', '.', ',']:
                                if len(self.tokenizer.tokenize(' ' + word)) == 1:
                                    word = self.tokenizer._convert_token_to_id(self.tokenizer.tokenize(' ' + word)[0])
                                else:
                                    word = self.tokenizer._convert_token_to_id(word)
                            else:
                                word = self.tokenizer._convert_token_to_id(word)
                            
                            if len(key) == 1:
                                label_to_word[self.label2id[int(key)]].append(word)
                            else:
                                label_to_word[self.label2id[key]].append(word)
            for key in self.label2id:
                label = self.label2id[key]
                k_map[label] = label_to_word[label][:top_k]
        
        self.model.label_token_list = {}

        for key in self.label2id:
            label = self.label2id[key]
            self.model.label_token_list[label] = torch.tensor(k_map[label]).long().cuda()
            
            print(label)
            print(self.tokenizer.decode(k_map[label]))
            
        return k_map
    
    def tokenize_multipart_input(
        self,
        input_text_list,
        max_length,
        prompt=False, 
        template=None,
        label_word_list=None,
        first_sent_limit=None,
        other_sent_limit=None,
        gpt3=False,
        truncate_head=False,
        support_labels=None,
    ):
        def enc(text):
            return self.tokenizer.encode(text, add_special_tokens=False)

        input_ids = []
        attention_mask = []
        token_type_ids = [] # Only for BERT
        mask_pos = None # Position of the mask token

        if prompt:
            """
            Concatenate all sentences and prompts based on the provided template.
            Template example: '*cls*It was*mask*.*sent_0**<sep>*label_0:*sent_1**<sep>**label_1*:*sent_2**<sep>*'
            *xx* represent variables:
                *cls*: cls_token
                *mask*: mask_token
                *sep*: sep_token
                *sep+*: sep_token, also means +1 for segment id
                *sent_i*: sentence i (input_text_list[i])
                *sent-_i*: same as above, but delete the last token
                *sentl_i*: same as above, but use lower case for the first word
                *sentl-_i*: same as above, but use lower case for the first word and delete the last token
                *+sent_i*: same as above, but add a space before the sentence
                *+sentl_i*: same as above, but add a space before the sentence and use lower case for the first word
                *label_i*: label_word_list[i]
                *label_x*: label depends on the example id (support_labels needed). this is only used in GPT-3's in-context learning

            Use "_" to replace space.
            PAY ATTENTION TO SPACE!! DO NOT leave space before variables, for this will lead to extra space token.
            """
            assert template is not None

            special_token_mapping = {
                'cls': self.tokenizer.cls_token_id, 'mask': self.tokenizer.mask_token_id, 'sep': self.tokenizer.sep_token_id, 'sep+': self.tokenizer.sep_token_id, 
            }
            template_list = template.split('*') # Get variable list in the template
            segment_id = 0 # Current segment id. Segment id +1 if encountering sep+.

            for part_id, part in enumerate(template_list):
                new_tokens = []
                segment_plus_1_flag = False
                if part in special_token_mapping:
                    if part == 'cls' and 'T5' in type(self.tokenizer).__name__:
                        # T5 does not have cls token
                        continue
                    new_tokens.append(special_token_mapping[part])
                    if part == 'sep+':
                        segment_plus_1_flag = True
                elif part[:6] == 'label_':
                    # Note that label_word_list already has extra space, so do not add more space ahead of it.
                    label_id = int(part.split('_')[1])
                    label_word = label_word_list[label_id]
                    new_tokens.append(label_word)
                elif part[:7] == 'labelx_':
                    instance_id = int(part.split('_')[1])
                    label_id = support_labels[instance_id]
                    label_word = label_word_list[label_id]
                    new_tokens.append(label_word)
                elif part[:5] == 'sent_':
                    sent_id = int(part.split('_')[1])
                    new_tokens += enc(input_text_list[sent_id]) 
                elif part[:6] == '+sent_':
                    # Add space
                    sent_id = int(part.split('_')[1])
                    new_tokens += enc(' ' + input_text_list[sent_id])
                elif part[:6] == 'sent-_':
                    # Delete the last token
                    sent_id = int(part.split('_')[1])
                    new_tokens += enc(input_text_list[sent_id][:-1])
                elif part[:6] == 'sentl_':
                    # Lower case the first token
                    sent_id = int(part.split('_')[1])
                    text = input_text_list[sent_id]
                    text = text[:1].lower() + text[1:]
                    new_tokens += enc(text)
                elif part[:7] == '+sentl_':
                    # Lower case the first token and add space 
                    sent_id = int(part.split('_')[1])
                    text = input_text_list[sent_id]
                    text = text[:1].lower() + text[1:]
                    new_tokens += enc(' ' + text)
                elif part[:7] == 'sentl-_':
                    # Lower case the first token and discard the last token
                    sent_id = int(part.split('_')[1])
                    text = input_text_list[sent_id]
                    text = text[:1].lower() + text[1:]
                    new_tokens += enc(text[:-1])
                elif part[:6] == 'sentu_':
                    # Upper case the first token
                    sent_id = int(part.split('_')[1])
                    text = input_text_list[sent_id]
                    text = text[:1].upper() + text[1:]
                    new_tokens += enc(text)
                elif part[:7] == '+sentu_':
                    # Upper case the first token and add space
                    sent_id = int(part.split('_')[1])
                    text = input_text_list[sent_id]
                    text = text[:1].upper() + text[1:]
                    new_tokens += enc(' ' + text)
                else:
                    # Just natural language prompt
                    part = part.replace('_', ' ') 
                    # handle special case when T5 self.tokenizer might add an extra space
                    if len(part) == 1:
                        new_tokens.append(self.tokenizer._convert_token_to_id(part))
                    else:
                        new_tokens += enc(part)

                if part[:4] == 'sent' or part[1:5] == 'sent':
                    # If this part is the sentence, limit the sentence length
                    sent_id = int(part.split('_')[1])
                    if sent_id == 0:
                        if first_sent_limit is not None:
                            new_tokens = new_tokens[:first_sent_limit]
                    else:
                        if other_sent_limit is not None:
                            new_tokens = new_tokens[:other_sent_limit]

                input_ids += new_tokens
                attention_mask += [1 for i in range(len(new_tokens))]
                token_type_ids += [segment_id for i in range(len(new_tokens))]

                if segment_plus_1_flag:
                    segment_id += 1
        else:
            input_ids = [self.tokenizer.cls_token_id]
            attention_mask = [1]
            token_type_ids = [0]

            for sent_id, input_text in enumerate(input_text_list):
                if input_text is None:
                    # Do not have text_b
                    continue
                if pd.isna(input_text) or input_text is None:
                    # Empty input
                    input_text = ''
                input_tokens = enc(input_text) + [self.tokenizer.sep_token_id]
                input_ids += input_tokens
                attention_mask += [1 for i in range(len(input_tokens))]
                token_type_ids += [sent_id for i in range(len(input_tokens))]

            if 'T5' in type(self.tokenizer).__name__: # T5 does not have CLS token
                input_ids = input_ids[1:]
                attention_mask = attention_mask[1:]
                token_type_ids = token_type_ids[1:]

        # Padding
        if first_sent_limit is not None and len(input_ids) > max_length:
            # If using sentence limit, the total length still exceeds the maximum limit, report a warning
            logger.warn("Input exceeds max_length limit: {}".format(self.tokenizer.decode(input_ids)))

        while len(input_ids) < max_length:
            input_ids.append(self.tokenizer.pad_token_id)
            attention_mask.append(0)
            token_type_ids.append(0)

        # Truncate
        if len(input_ids) > max_length:
            if truncate_head:
                input_ids = input_ids[-max_length:]
                attention_mask = attention_mask[-max_length:]
                token_type_ids = token_type_ids[-max_length:]
            else:
                # Default is to truncate the tail
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
                token_type_ids = token_type_ids[:max_length]

        # Find mask token
        if prompt:
            mask_pos = [input_ids.index(self.tokenizer.mask_token_id)]
            # Make sure that the masked position is inside the max_length
            assert mask_pos[0] < max_length

        result = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if 'BERT' in type(self.tokenizer).__name__:
            # Only provide token type ids for BERT
            result['token_type_ids'] = token_type_ids

        if prompt:
            result['mask_pos'] = mask_pos

        return result

    def preprocess(
        self,
        examples,
        input_key = ["sentence"], 
        label_key = "label", 
        max_length = 128, 
        template = "*cls**sent_0*_It_was*mask*.*sep+*",
        first_sent_limit=None,
        other_sent_limit=None,
    ):
        # Tokenize the texts
        result = {}
        result["input_ids"] = []
        result["attention_mask"] = []
        result["mask_pos"] = []

        if len(input_key) == 1:
            sentences = examples[input_key[0]]
            input_text_lists = [[sent] for sent in sentences]
        else:
            sentences1 = examples[input_key[0]]
            sentences2 = examples[input_key[1]]
            input_text_lists = [[sent1, sent2] for sent1, sent2 in zip(sentences1, sentences2)]
        
        for input_text_list in input_text_lists:
            res = self.tokenize_multipart_input(
                input_text_list=input_text_list,
                max_length=max_length,
                prompt=True,
                template=template,
                first_sent_limit=first_sent_limit,
                other_sent_limit=other_sent_limit,
                )
            
            result["input_ids"].append(res["input_ids"])
            result["attention_mask"].append(res["attention_mask"])
            result["mask_pos"].append(res["mask_pos"])

        if label_key in examples:
            result["labels"] = [self.label2id[label] for label in examples[label_key]]
        
        return result
    