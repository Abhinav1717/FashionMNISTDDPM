from torchvision.transforms.functional import to_tensor
from torchvision.transforms.functional import to_tensor,InterpolationMode,_interpolation_modes_from_int,resize
from torchvision.utils import _log_api_usage_once
from collections.abc import Sequence
import torch
import warnings

class toTensor(object):
    
    def __call__(self,sample):
        
        image = sample
        image_tensor = to_tensor(image)
        return (image_tensor)

class Normalize(object):
    def  __call__(self,sample):
        
        image = sample 
        image = (image - 0.5)*2
        return image
    
class Resize(torch.nn.Module):
    
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=None):
        super().__init__()
        _log_api_usage_once(self)
        if not isinstance(size, (int, Sequence)):
            raise TypeError(f"Size should be int or sequence. Got {type(size)}")
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")
        self.size = size
        self.max_size = max_size

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument 'interpolation' of type int is deprecated since 0.13 and will be removed in 0.15. "
                "Please use InterpolationMode enum."
            )
            interpolation = _interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, sample):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        return resize(sample, self.size, self.interpolation, self.max_size, self.antialias)
