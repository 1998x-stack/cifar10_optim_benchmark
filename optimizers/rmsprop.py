import torch.optim as optim

from .base import BaseOptimizerWrapper, register


@register("RMSprop")
class RMSpropWrapper(BaseOptimizerWrapper):
    name = "RMSprop"

    def build(self):
        return optim.RMSprop(
            self.params,
            lr=self.lr,
            alpha=self.kwargs.get("alpha", 0.99),
            eps=self.kwargs.get("eps", 1e-8),
            weight_decay=self.weight_decay,
            momentum=self.kwargs.get("momentum", 0.0),
            centered=self.kwargs.get("centered", False),
        )
