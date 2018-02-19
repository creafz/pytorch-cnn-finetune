import torch
from torch.autograd import Variable


# Polyfill that disables gradient calculation for Variable.
# Previous versions of PyTorch use volatile=True flag for variables
# and the latest version uses torch.no_grad context manager.

try:

    class no_grad_variable(torch.no_grad):

        def __init__(self, *args, **kwargs):
            super().__init__()
            self.variable = Variable(*args, **kwargs)

        def __enter__(self):
            super().__enter__()
            return self.variable

except AttributeError:

    class no_grad_variable:

        def __init__(self, *args, **kwargs):
            self.variable = Variable(*args, **kwargs, volatile=True)

        def __enter__(self):
            return self.variable

        def __exit__(self, *args, **kwargs):
            pass
