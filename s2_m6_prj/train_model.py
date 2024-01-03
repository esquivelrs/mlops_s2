import click
import torch
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
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--epochs", default=5, help="number of epochs to train for")
@click.option("--batch_size", default=256, help="batch size to use for training")
@click.option("--model_name", default="classifier", help="filename to save model as")
def train(lr, epochs, batch_size, model_name):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyNeuralNet.to(device)
    print(model)
    train_set = torch.load("data/processed/train_data.pt")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003)
    loss_list = []
    for epoch in range(epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} Loss {loss}")
        loss_list.append(loss.item())
    
    # create folder if does nor exist
    os.makedirs(f"models/{model_name}", exist_ok=True)
    os.makedirs(f"reports/figures/{model_name}", exist_ok=True)
    
    torch.save(model, f"models/{model_name}/model.pt")
    #save plot as png
    plt.plot(loss_list)
    plt.savefig(f"reports/figures/{model_name}/loss.png")


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    model = torch.load(model_checkpoint)
    test_set = torch.load("data/processed/test_data.pt")
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
    model.eval()

    test_preds = []
    test_labels = []
    with torch.no_grad():
        for x, y in test_dataloader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            test_preds.append(output.argmax(dim=1).cpu())
            test_labels.append(y.cpu())

    test_preds = torch.cat(test_preds, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    print((test_preds == test_labels).float().mean())


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
