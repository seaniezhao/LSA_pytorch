import torch
import torch.nn.functional as F

def concat_prev(bar_data, condition):
    """Concatenate conditioning vector on feature map axis."""
    if condition is None:
        return bar_data
    else:
        if bar_data.shape[2:4] == condition.shape[2:4]:

            return torch.cat((bar_data, condition), 1)
        else:
            raise ValueError('unmatched shape:', bar_data.shape, 'and', condition.shape)