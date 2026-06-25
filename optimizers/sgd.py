import torch.optim as optim

from .base import BaseOptimizerWrapper, register


@register("SGD")
class SGDWrapper(BaseOptimizerWrapper):
    name = "SGD"

    def build(self):
        return optim.SGD(
            self.params,
            lr=self.lr,
            momentum=self.kwargs.get("momentum", 0.9),
            dampening=self.kwargs.get("dampening", 0.0),
            weight_decay=self.weight_decay,
            nesterov=self.kwargs.get("nesterov", True),
        )
