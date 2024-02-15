from transformers import Trainer
from torch import nn
from transformers.utils import is_apex_available
import torch
import numpy as np


def sigmoid_focal_loss(logits, targets, alpha=-1, gamma= 2, reduction="mean"):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    BCLoss = nn.CrossEntropyLoss(reduction='none')
    ce_loss = BCLoss(logits, targets)
    p = torch.sigmoid(logits)
    p_t = p[np.arange(len(p)), targets]
    loss = ce_loss * ((1 - p_t) ** gamma)
    
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

def get_uniform_m_flip_mat(noise_level, num_classes=2):
    clean_label_probability = 1 - noise_level
    uniform_noise_probability = noise_level / (num_classes - 1)  # distribute noise_level across all other labels

    true_noise_matrix = np.empty((num_classes, num_classes))
    true_noise_matrix.fill(uniform_noise_probability)
    for true_label in range(num_classes):
        true_noise_matrix[true_label][true_label] = clean_label_probability

    return true_noise_matrix

class CustomTrainer(Trainer):
    def __init__(self, model, label_weights=None, noise_ratio=None, *args, **kwargs):
        super().__init__(model=model, *args, **kwargs)
        self.label_weights = label_weights  
        
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss using label_weights
        loss_fct = nn.CrossEntropyLoss(weight=self.label_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    



class CustomTrainerFocalLoss(Trainer):
    def __init__(self, model, alpha=-1, gamma=1, label_weights=None, noise_ratio=None, *args, **kwargs):
        super().__init__(model=model, *args, **kwargs)
        self.alpha = alpha
        self.gamma = gamma  
    
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # logits (torch.FloatTensor of shape (batch_size, config.num_labels)) — Classification (or regression if config.num_labels==1) scores (before SoftMax).

        # compute custom loss using label_weights
        loss = sigmoid_focal_loss(logits, labels, alpha=self.alpha, gamma=self.gamma)
        # loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class CustomTrainerWithConstantNoiseMatrix(Trainer):
    def __init__(self, model, label_weights=None, noise_ratio=0, *args, **kwargs):
        super().__init__(model=model, *args, **kwargs)
        self.label_weights = label_weights  
        self.noise_ratio = noise_ratio
        mat = get_uniform_m_flip_mat(noise_level=noise_ratio)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.noise_matrix = nn.Parameter(torch.tensor(mat).float(), requires_grad=False).to(device)
        self.logits2dist = nn.Softmax(dim=1)


    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)

        clean_logits = outputs.get("logits")
        clean_dist = self.logits2dist(clean_logits)
        noisy_prob = torch.matmul(clean_dist, self.noise_matrix)
        log_noisy_logits = torch.log(noisy_prob + 1e-6)
        # compute custom loss using label_weights
        loss_fct = nn.CrossEntropyLoss(weight=self.label_weights)
        loss = loss_fct(log_noisy_logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    

    def training_step(self, model: nn.Module, inputs) :
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # if is_sagemaker_mp_enabled():
        #     loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
        #     return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        # if self.use_apex:
        #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #         scaled_loss.backward()
        # else:
        self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps