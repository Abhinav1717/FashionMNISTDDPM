from torchvision.transforms.functional import to_tensor
class toTensor(object):
    
    def __call__(self,sample):
        
        image = sample
        image_tensor = to_tensor(image)
        return (image_tensor)