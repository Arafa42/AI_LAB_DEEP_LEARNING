import torch

from options.options import Options
from utilities.utils import plot_tensor, mse, init_pytorch, not_implemented


def create_image(options: Options) -> torch.Tensor:
    """ use options to put the tensor to the correct device. """

    # Solution 1 - using lists
    """
    red_channel = [[0.5021, 0.2843, 0.1935],
                   [0.8017, 0.5914, 0.7038]]
    blue_channel = [[0.1138, 0.0684, 0.5483],
                    [0.8733, 0.6004, 0.5983]]
    green_channel = [[0.9047, 0.6829, 0.3117],
                     [0.6258, 0.2893, 0.9914]]
                     
    rgb = [red_channel, blue_channel, green_channel]
    image = torch.FloatTensor(rgb)
    plot_tensor(image, "title")
    return image      
    """

    # Solution 2 - uses tensors, this solution works but warns that the code is slow
    ### UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow.
    ### Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
    ### (Triggered internally at  ..\torch\csrc\utils\tensor_new.cpp:201.)

    """
    red_channel = torch.tensor([
        [0.5021, 0.2843, 0.1935],
        [0.8017, 0.5914, 0.7038]
    ], device=options.device)

    blue_channel = torch.tensor([
        [0.1138, 0.0684, 0.5483],
        [0.8733, 0.6004, 0.5983]
    ], device=options.device)

    green_channel = torch.tensor([
        [0.9047, 0.6829, 0.3117],
        [0.6258, 0.2893, 0.9914]
    ], device=options.device)

    image = torch.tensor([red_channel.numpy(), blue_channel.numpy(), green_channel.numpy()])
    plot_tensor(image, "title")
    return image
   """

    # Solution 3 - uses tensors, removing the warning
    red_channel = torch.tensor([
        [0.5021, 0.2843, 0.1935],
        [0.8017, 0.5914, 0.7038]
    ], device=options.device)

    blue_channel = torch.tensor([
        [0.1138, 0.0684, 0.5483],
        [0.8733, 0.6004, 0.5983]
    ], device=options.device)

    green_channel = torch.tensor([
        [0.9047, 0.6829, 0.3117],
        [0.6258, 0.2893, 0.9914]
    ], device=options.device)

    red_channel = red_channel.unsqueeze(0)
    blue_channel = blue_channel.unsqueeze(0)
    green_channel = green_channel.unsqueeze(0)

    image = torch.concat([red_channel, blue_channel, green_channel], 0)
    plot_tensor(image, "title")
    return image


def lin_layer_forward(weights: torch.Tensor, random_image: torch.Tensor) -> torch.Tensor:
    print("weights:", weights)
    print("random_image:", random_image)

    multiplication = torch.multiply(weights, random_image)
    tot = torch.sum(multiplication)
    return torch.squeeze(torch.FloatTensor([tot]))


def tensor_network():
    target = torch.FloatTensor([0.5], device=options.device)
    print(f"The target is: {target.item():.2f}")
    plot_tensor(target, "Target")

    input_tensor = torch.FloatTensor([0.4, 0.8, 0.5, 0.3], device=options.device)
    weights = torch.FloatTensor([0.1, -0.5, 0.9, -1], device=options.device)

    """START TODO:  ensure that the tensor 'weights' saves the computational graph and the gradients after backprop"""
    weights.requires_grad()
    plot_tensor(weights.detach(), "new graph title")
    """END TODO"""

    # remember the activation a of a unit is calculated as follows:
    #      T
    # a = W * x, with W the weights and x the inputs of that unit
    output = lin_layer_forward(weights, input_tensor)
    print(f"Output value: {output.item(): .2f}")
    plot_tensor(output.detach(), "Initial Output")

    # We want a measure of how close we are according to our target
    loss = mse(output, target)
    print(f"The initial loss is: {loss.item():.2f}\n")

    # Lets update the weights now using our loss..
    print(f"The current weights are: {weights}")

    """START TODO: the loss needs to be backpropagated"""
    loss.requires_grad = True
    loss.backward()
    plot_tensor(loss.detach(), "Loss")
    """END TODO"""

    print(f"The gradients are: {weights.grad}")

    """START TODO: implement the update step with a learning rate of 0.5"""
    # use tensor operations, recall the following formula we've seen during class: x <- x - alpha * x'
    weights = torch.subtract(weights, torch.multiply(weights, 0.5))
    """END TODO"""

    print(f"The new weights are: {weights}\n")

    # What happens if we forward through our layer again?
    output = lin_layer_forward(weights, input_tensor)
    print(f"Output value: {output.item(): .2f}")
    plot_tensor(output.detach(), "Improved Output")


if __name__ == "__main__":
    options = Options()
    init_pytorch(options)
    tensor_network()
