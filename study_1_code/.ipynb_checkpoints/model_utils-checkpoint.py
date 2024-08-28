import torch
import torch.nn.functional as F
import torch.nn as nn




def update_learning_rate(optimizer, new_lr):
    """
    Update the learning rate of the given optimizer.

    Parameters:
    optimizer (torch.optim.Optimizer): The optimizer whose learning rate needs to be changed.
    new_lr (float): The new learning rate to be set.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def logits_to_continuous_label(logits):
    """
    Convert logits to a continuous class label.

    Args:
    logits (torch.Tensor): The logits output from the model. Shape (batch_size, num_classes)
    
    Returns:
    torch.Tensor: Continuous class labels. Shape (batch_size,)
    """
    # Convert logits to probabilities
    probabilities = F.softmax(logits, dim=1)
    
    # Create a tensor of class indices (0, 1, 2, ..., num_classes-1)
    class_indices = torch.arange(logits.size(1), dtype=torch.float, device=logits.device)
    
    # Calculate the expected value
    continuous_labels = torch.sum(probabilities * class_indices, dim=1)
    
    return continuous_labels




# Function to check the training device
def check_device():
    if torch.cuda.is_available():
        print(f"Training on GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        return torch.device("cuda")
    else:
        print("Training on CPU")
        return torch.device("cpu")


