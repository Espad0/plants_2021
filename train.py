from data_processing import build_sampler, CustomDataset, ToTensor, Rescale
from training_utils import run_training
from torch.utils.data import DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
import torch

train_csv_path = 'train_dataset.csv'
val_csv_path = 'val_dataset.csv'
savepath = 'model_30epoch_balanced_classes.pt'
img_dir = '../resized_train_images/'
epochs = 30

if __name__ == '__main__':
    transformations=transforms.Compose([
        Rescale((224,224)),
        ToTensor()
    ])

    train_dataset = CustomDataset(csv_file=train_csv_path,
                                img_dir=img_dir,
                                transform=transformations)

    val_dataset = CustomDataset(csv_file=val_csv_path,
                                img_dir=img_dir,
                                transform=transformations)

    train_sampler = build_sampler(train_csv_path)

    train_dataloader = DataLoader(train_dataset, batch_size=4,
                                sampler=train_sampler)

    val_dataloader = DataLoader(val_dataset, batch_size=4,
                            shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vgg16 = models.vgg16(pretrained=True)
    vgg16.classifier[6].out_features = 12
    # freeze convolution weights
    for param in vgg16.features.parameters():
        param.requires_grad = False

    # optimizer
    optimizer = optim.SGD(vgg16.classifier.parameters(), lr=0.001, momentum=0.9)
    # loss function
    criterion = nn.CrossEntropyLoss()

    vgg16 = vgg16.to(device)

    run_training(vgg16, savepath, epochs, train_dataloader, val_dataloader,
                    criterion, optimizer)