from .madrylab import MadryLabCIFARModel,MadryLabMNISTModel




def get_model(model_name):
    return eval(model_name)()