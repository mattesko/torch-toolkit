import torch
import torch.nn.functional as F

from metrics import dice_score


def train_one_epoch(net, dataloader, optimizer, criterion, 
                    device=None, num_accumulated_steps=1,  
                    input_dtype=torch.double,
                    target_dtype=torch.long):

    """Train the network on the entire dataset once
    Args:
        net (torch.nn.Module): the neural network to train
        dataloader (torch.utils.data.Dataloader): the batched data to train on
        optimizer (torch.optim): the optimizer for the network (Adam, etc.)
        criterion (callable): the loss function. Expects pairs of outputs and
            targets as inputs
        device (torch.device, optional): the device to move data on
        input_dtype (torch.dtype): the input's data type to convert to if a 
            device is used
        target_dtype (torch.dtype): the target's data type to convert to if a 
            device is used
        num_accumulated_steps (int): the number of times the loss gradients are
            accumulated before a step in the optimizer is taken.
            Helpful to increase this to speed up training if your batch size is
            low (default: 1)

    Returns:
        network (torch.nn.Module), running_loss (float32)
    """
    net.train()
    torch.set_grad_enabled(True)
    optimizer.zero_grad()
    running_loss = 0.0

    if criterion.__name__ == 'dice_loss': use_dice_loss = False

    for i, data in enumerate(dataloader):

        input_images, targets = data
        
        if device:
            input_images = input_images.to(device, input_dtype)
            targets = targets.to(device, target_dtype)
        
        outputs = net(input_images)

        # TODO add dice loss functionality
        # if use_dice_loss:
        #     outputs = F.log_softmax(outputs, dim=1)
        #     outputs = outputs.unsqueeze(dim=1)
        #     loss = criterion(outputs, targets)
        # else:
        targets = targets.squeeze(dim=1)
        loss = criterion(outputs, targets)

        loss.backward()
        running_loss += loss.detach().cpu().numpy()

        if (i+1) % num_accumulated_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            running_loss /= num_accumulated_steps

    if (i+1) % num_accumulated_steps != 0:
        optimizer.step()
        running_loss /= (num_accumulated_steps - i % num_accumulated_steps)

    running_loss = running_loss.item()
    return net, running_loss


def validate(net, dataloader, metric=dice_score, device=torch.device('cpu'), 
             input_dtype=torch.double, target_dtype=torch.long, 
             use_dice_loss=False, threshold=0.5):
    """
    Gather validation segmentation metrics (Dice, Jaccard) on neural network
    
    Arguments:
        net (torch.nn.Module): the neural network to validate
        dataloader (torch.utils.data.DataLoader): validation data
        device (torch.device, optional): the device to move data on
        input_dtype (torch.dtype): the input's data type to convert to if a 
            device is used
        target_dtype (torch.dtype): the target's data type to convert to if a 
            device is used
        use_dice_loss

    """
    net.eval()
    torch.set_grad_enabled(False)
    score_mean = torch.zeros((1), device=device)
    jaccard_mean = torch.zeros((1), device=device)

    for i, data in enumerate(dataloader):

        input_images, targets = data
        
        if device:
            input_images = input_images.to(device, input_dtype)
            targets = targets.to(device, target_dtype)
        
        outputs = net(input_images)

        # TODO add dice loss functionality
        # if use_dice_loss:
        #     outputs = F.log_softmax(outputs, dim=1)
        # else:
        outputs = F.softmax(outputs, dim=1)
        
        # Ignore the background
        outputs = F.threshold(outputs[:, 1:, :, :], threshold, 0)
        outputs = torch.round(outputs)

        score = metric(outputs, targets)
        score = score.detach().cpu()
        score_mean = score_mean + (score - score_mean) / (i + 1)

    return score_mean.item()