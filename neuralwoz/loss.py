import torch


class LabelSmoothingLoss(torch.nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_size, pad_id=1, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()
        self.eps = label_smoothing
        self.n_class = tgt_size
        self.ignore_index = ignore_index
        self.pad_id = pad_id
    
    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        mask = target.ne(self.ignore_index).float()
        target_ = target.masked_fill(target.eq(self.ignore_index), self.pad_id)
        one_hot = torch.zeros_like(output).scatter(1, target_.view(-1, 1), 1)
        one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / (self.n_class - 1)
        one_hot = one_hot.to(target.device)
        log_prb = torch.nn.functional.log_softmax(output, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss * mask
        return loss.mean()
    
    
class LabelSmoothingWithoutSoftmaxLoss(torch.nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_size, pad_id=1, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        super(LabelSmoothingWithoutSoftmaxLoss, self).__init__()
        self.eps = label_smoothing
        self.n_class = tgt_size
        self.ignore_index = ignore_index
        self.pad_id = pad_id
    
    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        mask = target.ne(self.ignore_index).float()
        target_ = target.masked_fill(target.eq(self.ignore_index), self.pad_id)
        one_hot = torch.zeros_like(output).scatter(1, target_.view(-1, 1), 1)
        one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / (self.n_class - 1)
        one_hot = one_hot.to(target.device)
        log_prb = torch.log(output)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss * mask
        return loss.mean()

    
class CrossEntropyWithoutSoftmaxLoss(torch.nn.Module):

    def __init__(self, tgt_size, pad_id=1, ignore_index=-100):
        super(CrossEntropyWithoutSoftmaxLoss, self).__init__()
        self.n_class = tgt_size
        self.ignore_index = ignore_index
        self.pad_id = pad_id
    
    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        mask = target.ne(self.ignore_index).float()
        target_ = target.masked_fill(target.eq(self.ignore_index), self.pad_id)
        one_hot = torch.zeros_like(output).scatter(1, target_.view(-1, 1), 1)
        one_hot = one_hot.to(target.device)
        log_prb = torch.log(output)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss * mask
        return loss.mean()
    
    
class WeightedChoiceLoss(torch.nn.Module):
    def __init__(self, beta):
        super(WeightedChoiceLoss, self).__init__()
        self.beta = beta
    
    def forward(self, output, target, weight):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        weight (FloatTensor): batch_size (0. or 1.)
        """

        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot.to(target.device)
        log_prb = torch.nn.functional.log_softmax(output, dim=1)
        loss = -(one_hot * log_prb).sum(1)
        loss = self.beta * (loss * weight) + (1 - weight) * loss
        
        return loss.mean()
