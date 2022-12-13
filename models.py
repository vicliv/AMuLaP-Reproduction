import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers.modeling_utils import PreTrainedModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertOnlyMLMHead
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead, RobertaClassificationHead
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2PreTrainedModel, DebertaV2Model
from transformers.models.opt.modeling_opt import OPTPreTrainedModel, OPTModel
from transformers import AutoModelWithLMHead, AutoModel

import logging
logger = logging.getLogger(__name__)

def forward_function(
    model, 
    lm_head = None,
    label_token_list = None, 
    input_ids=None,
    attention_mask=None,
    mask_pos=None,
    labels=None,
):
    batch_size = input_ids.size(0)

    if mask_pos is not None:
        mask_pos = mask_pos.squeeze()

    # Encode everything
    outputs = model(
        input_ids,
        attention_mask=attention_mask
    )

    # Get <mask> token representation
    sequence_output = outputs[0]
    
    sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

    # Logits over vocabulary tokens
    if lm_head is not None:
        prediction_mask_scores = lm_head(sequence_mask_output)
    else:
        prediction_mask_scores =  sequence_mask_output
    
#         all_logits = prediction_mask_scores
    all_logits = F.softmax(prediction_mask_scores, dim=-1)
    

    if label_token_list is not None:
        logits = []
        for label in label_token_list:
            logits.append(torch.sum(all_logits[:, label_token_list[label]], 1).unsqueeze(-1))
        logits = torch.cat(logits, -1)

    loss = None
    if labels is not None:
#             loss_fct = nn.CrossEntropyLoss()
        loss_fct = nn.NLLLoss()
#             loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss = loss_fct(torch.log(logits.view(-1, logits.size(-1))), labels.view(-1))

    output = (all_logits,)
    if label_token_list is not None:
        output = ((logits,) + output)
    return ((loss,) + output) if loss is not None else output
        

class RobertaForPromptFinetuning(RobertaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.init_weights()

        self.label_token_list = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
    ):
        return forward_function(self.roberta, self.lm_head, self.label_token_list, input_ids, attention_mask, mask_pos, labels)
    
    

class AutoModelForPromptFinetuning(PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        
        self._init_weights = self.model._init_weights
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        
        self.init_weights()

        self.label_token_list = None


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
    ):
        return forward_function(self.model, self.lm_head, self.label_token_list, input_ids, attention_mask, mask_pos, labels)


class BertForPromptFinetuning(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

        self.label_token_list = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
    ):
        return forward_function(self.bert, self.cls, self.label_token_list, input_ids, attention_mask, mask_pos, labels)


class DebertaV2ForPromptFinetuning(DebertaV2PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.deberta = DebertaV2Model(config)
        self.lm_predictions = transformers.models.deberta_v2.modeling_deberta_v2.NewDebertaV2OnlyMLMHead(config)
        self.post_init()

        self.label_token_list = None
    
    def resize_token_embeddings(self, new_num_tokens: int):

        old_bias = self.lm_predictions.lm_head.bias.data

        new_bias = nn.Parameter(torch.zeros(new_num_tokens))

        # If there are more tokens, will transfer all of old values
        # If there are fewer tokens, will transfer some of old values
        num_to_transfer = min(len(old_bias), new_num_tokens)
        new_bias.data[:num_to_transfer] = old_bias[:num_to_transfer]

        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)

        self.lm_predictions.lm_head.bias = new_bias

        return self.get_input_embeddings()

    def get_output_embeddings(self):
        return None

    def set_output_embeddings(self, new_embeddings):
        return None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
    ):
       return forward_function(self.deberta, lambda x: self.lm_predictions(x, self.deberta.embeddings.word_embeddings), self.label_token_list, input_ids, attention_mask, mask_pos, labels)
    

class OPTForPromptFinetuning(OPTPreTrainedModel):
    # THIS MODEL DOES NOT WORK SINCE IT WAS NOT TRAINED FOR MASK MODELING
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.opt = OPTModel(config)
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)
        self.init_weights()

        self.label_token_list = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
    ):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        outputs = self.opt(
            input_ids,
            attention_mask=attention_mask
        )

        # Get <mask> token representation
        sequence_output = outputs[0]
        
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.lm_head(sequence_mask_output)
        
#         all_logits = prediction_mask_scores
        all_logits = F.softmax(prediction_mask_scores, dim=-1)
        

        if self.label_token_list is not None:
            logits = []
            for label in self.label_token_list:
                logits.append(torch.sum(all_logits[:, self.label_token_list[label]], 1).unsqueeze(-1))
            logits = torch.cat(logits, -1)

        loss = None
        if labels is not None:
#             loss_fct = nn.CrossEntropyLoss()
            loss_fct = nn.NLLLoss()
#             loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = loss_fct(torch.log(logits.view(-1, logits.size(-1))), labels.view(-1))

        output = (all_logits,)
        if self.label_token_list is not None:
            output = ((logits,) + output)
        return ((loss,) + output) if loss is not None else output