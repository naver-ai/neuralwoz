"""
NeuralWOZ
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
"""

from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import RobertaTokenizer, RobertaForMultipleChoice
import torch
import torch.nn.functional as F
from loss import LabelSmoothingLoss, WeightedChoiceLoss


class BartForConditionalGeneration(BartForConditionalGeneration):
    def __init__(self, config, lsm_f=0.1):
        super(BartForConditionalGeneration, self).__init__(config)
        self.lsm_f = lsm_f  # label smoothing
        self.pad_id = config.pad_token_id
        
    def forward(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_cached_states=None,
        lm_labels=None,
        use_cache=False,
        **unused
    ):

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
        )
        
        lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        outputs = (lm_logits,) + outputs[1:] # Add cache, hidden states and attention if they are here
        if lm_labels is not None:
            if self.lsm_f > 0.:
                loss_fct = LabelSmoothingLoss(self.lsm_f, self.config.vocab_size, self.pad_id)
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
            # TODO(SS): do we need to ignore pad tokens in lm_labels?
            lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), lm_labels.view(-1))
            outputs = (lm_loss,) + outputs

        return outputs

    
class RobertaForMultipleChoice(RobertaForMultipleChoice):

    def __init__(self, config, beta=5.0):
        super().__init__(config)
        self.beta = beta

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        not_none_mask=None,
        target_masks=None
    ):
    
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        outputs = self.roberta(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)
        if target_masks is not None:
            reshaped_logits = reshaped_logits.masked_fill(target_masks.ne(1), -1e3)
        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.beta > 0. and not_none_mask is not None:
                loss_fct = WeightedChoiceLoss(self.beta)
                loss = loss_fct(reshaped_logits, labels, not_none_mask)
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)
