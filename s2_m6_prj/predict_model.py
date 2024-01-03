import torch
import click
import os

from models.model import MyNeuralNet

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.argument("model_checkpoint")
@click.argument("predict_image")
def predict(
    model_checkpoint: str,
    predict_image: str,
) -> None:
    """Run prediction for a given model and dataloader.
    
    Args:
        model: model to use for prediction
        dataloader: dataloader with batches
    
    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    model = torch.load(model_checkpoint)
    #get only one item from the dataset
    
    image = torch.load(predict_image)[0][0]
    image = image.unsqueeze(0)
    print(image.shape)
    
    model.eval()

    with torch.no_grad():
        x = image
        x = x.to(device)
        output = model(x)
        #print classification results
        print(output.argmax(dim=1).cpu())


cli.add_command(predict)


if __name__ == "__main__":
    cli()
