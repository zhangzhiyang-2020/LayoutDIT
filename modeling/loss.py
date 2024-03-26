import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from focal_loss.focal_loss import FocalLoss

class FocalLoss_(_Loss):
    
    def __init__(self, num_label, ignore_index, size_average=None, reduce=None, reduction='mean'):
        super(FocalLoss_, self).__init__(size_average=size_average, reduce=reduce, reduction=reduction)
        self.num_label = num_label
        self.ignore_index = ignore_index
        # self.label_smoothing = label_smoothing
        # self.confidence = 1.0 - label_smoothing

        self.focal_loss = FocalLoss(gamma=0.7, reduction="none")

    @staticmethod
    def loss_mask_and_normalize(loss, mask):
            """
                loss: shape = [batch_size, seq_len].
                mask: shape = [batch_size, seq_len].
            """
            mask = mask.type_as(loss)
            loss = loss * mask
            denominator = torch.sum(mask, dim=-1) + 1e-5 # shape = [batch_size].
            loss_each_instance = torch.sum(loss, dim=-1) / denominator # shape = [batch_size].
            avg_loss = torch.mean(loss_each_instance) # shape = [].
            # denominator = torch.sum(mask) + 1e-5
            # avg_loss = (loss / denominator).sum()
            return avg_loss
    
    def forward(self, model_output, target, loss_mask):
        """
            model_output | tensor of shape = [batch_size, max_tgt_len, num_label], after softmax operation. dtype = float16 or float32.
            target | tensor of shape = [batch_size, max_tgt_len] with each element being in the range of [0, num_label - 1]. dtype = int32.
        """

        batch_size, max_tgt_len, num_label = model_output.shape

        # smoothing_value = torch.tensor(self.label_smoothing / (self.num_label - 1), dtype=model_output.dtype, device=model_output.device)
        # soft_target = smoothing_value.repeat((batch_size, max_tgt_len, num_label)) # shape = [batch_size, max_tgt_len, num_label].

        # target = target.view(-1) # shape = [batch_size, * max_tgt_len].
        # soft_target = soft_target.reshape((-1, num_label)) # shape = [batch_size * max_tgt_len, num_label].
        # soft_target.scatter_(1, target.unsqueeze(1), self.confidence) # shape = [batch_size * max_tgt_len, num_label].

        # model_output = model_output.view(-1, num_label) # shape = [batch_size * max_tgt_len, num_label].
        
        # loss = self.focal_loss(model_output, soft_target).reshape((batch_size, max_tgt_len)) # shape = [batch_size, max_tgt_len].

        loss = self.focal_loss(model_output, target).reshape((batch_size, max_tgt_len)) # shape = [batch_size, max_tgt_len].

        masked_loss = self.loss_mask_and_normalize(loss.float(), loss_mask)

        return masked_loss



class CrossEntropyLoss_(_Loss):

    def __init__(self, label_smoothing, max_src_len, ignore_index, size_average=None, reduce=None, reduction='mean'):
        super(CrossEntropyLoss_, self).__init__(size_average=size_average, reduce=reduce, reduction=reduction)
        
        assert 0.0 <= label_smoothing <= 1.0

        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.confidence = 1.0 - label_smoothing

        self.max_src_len = max_src_len

    @staticmethod
    def loss_mask_and_normalize(loss, mask):
            """
                loss: shape = [batch_size, seq_len].
                mask: shape = [batch_size, seq_len].
            """
            mask = mask.type_as(loss)
            loss = loss * mask
            denominator = torch.sum(mask, dim=-1) + 1e-5 # shape = [batch_size].
            loss_each_instance = torch.sum(loss, dim=-1) / denominator # shape = [batch_size].
            avg_loss = torch.mean(loss_each_instance) # shape = [].
            # denominator = torch.sum(mask) + 1e-5
            # avg_loss = (loss / denominator).sum()
            return avg_loss

    def forward(self, model_output, target, num_effective_src_tokens, loss_mask):
        """
            model_output | tensor of shape = [batch_size, max_tgt_len, max_src_len], after log-softmax operation. dtype = float16 or float32.
            target | tensor of shape = [batch_size, max_tgt_len] with each element being in the range of [0, max_src_len - 1]. dtype = int32.
            num_effective_src_tokens | tensor of shape = [batch_size]. dtype = int32.
        """

        batch_size, max_tgt_len, max_src_len = model_output.shape
        assert max_src_len == self.max_src_len

        smoothing_value = self.label_smoothing / (num_effective_src_tokens + 1 - 1).to(model_output.dtype) # shape = [batch_size]. "+ 1" for cls_token_index prediction, "- 1" for the gt token_index exclusion.
        smoothing_value = smoothing_value.repeat(max_tgt_len, 1).permute(1, 0) # shape = [batch_size, max_tgt_len].

        # num_effective_src_tokens = torch.tensor([2, 5, 3, 7, 4])
        smoothing_value_mask = []
        for i, num_effective_src_token in enumerate(num_effective_src_tokens):
            smoothing_value_mask.append([1.0] + [1.0] * num_effective_src_token + [0.0] * (max_src_len - num_effective_src_token - 1)) # CLS_token should be taken into consideration.

        smoothing_value_mask = torch.tensor(smoothing_value_mask, dtype=model_output.dtype, device=model_output.device) # shape = [batch_size, max_src_len].

        soft_target = smoothing_value.unsqueeze(2) * smoothing_value_mask.unsqueeze(1) # shape = [batch_size, max_tgt_len, max_src_len].


        target = target.view(-1) # shape = [batch_size * max_tgt_len].
        soft_target = soft_target.reshape((-1, max_src_len)) # shape = [batch_size * max_tgt_len, max_src_len].
        soft_target.scatter_(1, target.unsqueeze(1), self.confidence) # shape = [batch_size * max_tgt_len, max_src_len].

        model_output = model_output.view(-1, max_src_len) # for dim compatible. shape = [batch_size * max_tgt_len, max_src_len].

        loss = F.kl_div(model_output, soft_target, reduction='none').view(batch_size, max_tgt_len, -1).sum(2) # shape = [batch_size, max_tgt_len].

        masked_loss = self.loss_mask_and_normalize(
            loss.float(), loss_mask)

        return masked_loss


class CrossEntropyLossForTrans_(_Loss):


    def __init__(self, label_smoothing, vocab_size, ignore_index, size_average=None, reduce=None, reduction='mean'):
        super(CrossEntropyLossForTrans_, self).__init__(size_average=size_average, reduce=reduce, reduction=reduction)
        
        assert 0.0 <= label_smoothing <= 1.0

        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.confidence = 1.0 - label_smoothing

        self.vocab_size = vocab_size

    @staticmethod
    def loss_mask_and_normalize(loss, mask):
            """
                loss: shape = [batch_size, seq_len].
                mask: shape = [batch_size, seq_len].
            """
            mask = mask.type_as(loss)
            loss = loss * mask
            denominator = torch.sum(mask, dim=-1) + 1e-7 # shape = [batch_size].
            loss_each_instance = torch.sum(loss, dim=-1) / denominator # shape = [batch_size].
            avg_loss = torch.mean(loss_each_instance) # shape = [].
#             denominator = torch.sum(mask) + 1e-5
#             avg_loss = (loss / denominator).sum()
            return avg_loss

    def forward(self, model_output, target, loss_mask):
        """
            model_output | tensor of shape = [batch_size, max_tgt_len, vocab_size], after log-softmax operation. dtype = float16 or float32.
            target | tensor of shape = [batch_size, max_tgt_len] with each element being in the range of [0, vocab_size - 1]. dtype = int32.
        """

        batch_size, max_tgt_len, vocab_size = model_output.shape
        assert vocab_size == self.vocab_size

        smoothing_value = self.label_smoothing / vocab_size
        smoothing_value = torch.ones((batch_size, max_tgt_len), device=model_output.device) * smoothing_value # shape = [batch_size, max_tgt_len].
        # smoothing_value = smoothing_value.repeat(max_tgt_len, 1).permute(1, 0) # shape = [batch_size, max_tgt_len].

        # # num_effective_src_tokens = torch.tensor([2, 5, 3, 7, 4])
        # smoothing_value_mask = []
        # for i, num_effective_src_token in enumerate(num_effective_src_tokens):
        #     smoothing_value_mask.append([1.0] + [1.0] * num_effective_src_token + [0.0] * (max_src_len - num_effective_src_token - 1)) # CLS_token should be taken into consideration.

        # smoothing_value_mask = torch.tensor(smoothing_value_mask, dtype=model_output.dtype).to(smoothing_value.device) # shape = [batch_size, max_src_len].

        soft_target = smoothing_value.unsqueeze(2).expand((batch_size, max_tgt_len, vocab_size)).clone() # shape = [batch_size, max_tgt_len, vocab_size].

        # print(target.shape)
        # print(soft_target.shape)
        # target = target.view(-1) # shape = [batch_size * max_tgt_len].
        target = target.reshape(-1)
        soft_target = soft_target.reshape((-1, vocab_size)) # shape = [batch_size * max_tgt_len, vocab_size].
        soft_target.scatter_(1, target.unsqueeze(1), self.confidence) # shape = [batch_size * max_tgt_len, vocab_size].

        model_output = model_output.view(-1, vocab_size) # for dim compatible. shape = [batch_size * max_tgt_len, vocab_size].

        loss = F.kl_div(model_output, soft_target, reduction='none').view(batch_size, max_tgt_len, -1).sum(2) # shape = [batch_size, max_tgt_len].

        masked_loss = self.loss_mask_and_normalize(
            loss.float(), loss_mask)

        return masked_loss