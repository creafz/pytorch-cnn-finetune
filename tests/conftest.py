import pytest
import torch

from cnn_finetune.shims import no_grad_variable


@pytest.fixture(scope='function', params=[(1, 3, 224, 224)])
def input_var(request):
    size = request.param
    torch.manual_seed(42)
    with no_grad_variable(torch.rand(size)) as var:
        yield var
