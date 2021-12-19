from .madrylab import MadryLabCIFARModel,MadryLabMNISTModel
from .resnet import resnet18, resnet34, resnet50



def get_model(model_name):
    return eval(model_name)()
