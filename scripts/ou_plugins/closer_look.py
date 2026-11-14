"""ME+GD and IDK+AP (Yuan et al., ICLR 2025, "A Closer Look at Machine Unlearning for LLMs") as
OpenUnlearning trainers. Losses follow sail-sg/closer-look-LLM-unlearning trainer/losses.py:
  ME+GD : forget_coeff * KL(uniform || p) on forget tokens (F.kl_div(log p, uniform)) + reg_coeff * NLL(retain)
  IDK+AP: forget_coeff * NLL(forget question -> IDK answer)
          + reg_coeff * AP, AP = -log sigmoid(beta * (NLL(retain idk) - NLL(retain answer))) / beta
The official ME+GD run uses mask=false (all non-pad tokens of the forget sample); IDK+AP uses mask=true.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainer.unlearn.grad_diff import GradDiff


def _xy(d):
    return {"input_ids": d["input_ids"], "attention_mask": d["attention_mask"], "labels": d["labels"]}


def _batch_nll(logits, labels):
    """Summed token NLL per sequence (get_batch_loss in the official code)."""
    loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    return loss(logits[..., :-1, :].transpose(-1, -2), labels[..., 1:]).sum(-1)


class MEGD(GradDiff):
    def __init__(self, me_mask=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.me_mask = me_mask

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        f = inputs["forget"]
        out = model(input_ids=f["input_ids"], attention_mask=f["attention_mask"])
        logits = out.logits[:, :-1, :]
        mask = (f["labels"][:, 1:] != -100) if self.me_mask else f["attention_mask"][:, 1:].bool()
        logp = F.log_softmax(logits.float(), -1)
        # KL(uniform || p) = -log V - mean_v log p_v, as F.kl_div(log p, uniform).sum(-1) in the official code
        kl = -logp.mean(-1) - torch.log(torch.tensor(float(logits.shape[-1]), device=logits.device))
        forget_loss = (kl * mask).sum() / mask.sum()
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=_xy(inputs["retain"]))
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, out) if return_outputs else loss


class IDKAP(GradDiff):
    def __init__(self, beta=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        out = model(**_xy(inputs["forget"]["alternate"]))
        forget_loss = out.loss
        r, r_idk = _xy(inputs["retain"]["original"]), _xy(inputs["retain"]["alternate"])
        nll = _batch_nll(model(**r).logits, r["labels"])
        nll_idk = _batch_nll(model(**r_idk).logits, r_idk["labels"])
        ap = -F.logsigmoid(self.beta * (nll_idk - nll)).mean() / self.beta
        loss = self.gamma * forget_loss + self.alpha * ap
        return (loss, out) if return_outputs else loss
