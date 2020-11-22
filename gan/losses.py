import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss

from torch.autograd import Variable     #1
dtype = torch.FloatTensor       #1

'''
def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.

    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

    Inputs:
    - input: PyTorch Variable of shape (N, ) giving scores.
    - target: PyTorch Variable of shape (N,) containing 0 and 1 giving targets.

    Returns:
    - A PyTorch Variable containing the mean BCE loss over the minibatch of input data.
    """
    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()
'''

def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss.
    
    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits 
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.
    
    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """

    # Batch size.
    n = logits_real.size()

    # Target label vector, the discriminator should be aiming
    true_labels = Variable(torch.ones(n)).type(dtype)

    # Discriminator loss has 2 parts: how well it classifies real images and how well it
    # classifies fake images.
    real_image_loss = bce_loss(logits_real, true_labels)
    fake_image_loss = bce_loss(logits_fake, 1 - true_labels)        # one-hot labels

    loss = real_image_loss + fake_image_loss

    
    ##########       END      ##########
    
    return loss

def generator_loss(logits_fake):
    """
    Computes the generator loss.
    
    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits 
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    
    #loss = None
    # Batch size.
    n = logits_fake.size()

    # Generator is trying to make the discriminator output 1 for all its images.
    # Discriminator determines if the image is real or fake.(1 or 0)
    # So we create a 'target' label vector of ones for computing generator loss.
    true_labels = Variable(torch.ones(n)).type(dtype)

    # Compute the generator loss compraing
    loss = bce_loss(logits_fake, true_labels)

    
    
    ##########       END      ##########
    
    return loss


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.
    
    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    
    loss = None
    
    ####################################
    n, _ = scores_real.size()
    loss_real = 0.5 * torch.mean(torch.pow(scores_real - Variable(torch.ones(n)).type(dtype), 2))
    loss_fake = 0.5 * torch.mean(torch.pow(scores_fake, 2))
    loss = loss_real + loss_fake

    ####################################
    
    
    ##########       END      ##########
    
    return loss

def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.
    
    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    
    loss = None
    
    ####################################
    n, _ = scores_fake.size()

    # Generator is trying to make the discriminator output 1 for all its images.
    # Discriminator determines if the image is real or fake.(1 or 0)
    # So we create a 'target' label vector of ones for computing generator loss.
    true_labels = Variable(torch.ones(n)).type(dtype)

    # Compute the generator loss compraing
    loss = 0.5 * torch.mean(torch.pow(scores_fake - Variable(torch.ones(n)).type(dtype), 2))
    ####################################
    
    
    ##########       END      ##########
    
    return loss
