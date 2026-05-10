from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_video_dataloaders(train_dir, test_dir, batch_size=32):

    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((48,48)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((48,48)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=train_transform
    )

    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader