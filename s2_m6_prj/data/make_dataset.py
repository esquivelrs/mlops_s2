
import torch


def main():
    train_data, train_labels = [ ], [ ]
    for i in range(5):
        train_data.append(torch.load(f"data/raw/train_images_{i}.pt"))
        train_labels.append(torch.load(f"data/raw/train_target_{i}.pt"))


    train_data = torch.cat(train_data, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    test_data = torch.load("data/raw/test_images.pt")
    test_labels = torch.load("data/raw/test_target.pt")
    
    
    train_data = torch.nn.functional.normalize(train_data, dim=1)
    test_data = torch.nn.functional.normalize(test_data, dim=1)

    train_data = train_data.unsqueeze(1)
    test_data = test_data.unsqueeze(1)
    
    train_data = torch.utils.data.TensorDataset(train_data, train_labels)
    test_data = torch.utils.data.TensorDataset(test_data, test_labels)
    
    # normalise and standardize the data
    print("Normalised data")
    torch.save(train_data, "data/processed/train_data.pt")
    torch.save(test_data, "data/processed/test_data.pt")
    
    
if __name__ == '__main__':
    # Get the data and process it
    main()