import torch.optim as optim

from .base import BaseOptimizerWrapper, register


@register("Adam")
class AdamWrapper(BaseOptimizerWrapper):
    name = "Adam"

    def build(self):
        return optim.Adam(
            self.params,
            lr=self.lr,
            betas=self.kwargs.get("betas", (0.9, 0.999)),
            eps=self.kwargs.get("eps", 1e-8),
            weight_decay=self.weight_decay,
            amsgrad=self.kwargs.get("amsgrad", False),
        )
