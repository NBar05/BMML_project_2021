from .cifar import get_cifar_dataset, get_mnist_dataset




def get_dataset(dataset_name):
    if dataset_name == 'CIFAR10':
        return get_cifar_dataset()
    elif dataset_name == 'MNIST':
        return get_mnist_dataset()
    else:
        return NotImplemented()