import torch
from tqdm import tqdm
import numpy as np
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# validation function
def validate(model, test_dataloader, criterion):
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    for int, data in enumerate(test_dataloader):
        data, target = data['image'].to(device, dtype=torch.float), data['label'].to(device)
        output = model(data)
        loss = criterion(output, target)
        
        val_running_loss += loss.item()
        _, preds = torch.max(output.data, 1)
        val_running_correct += (preds == target).sum().item()
    
    val_loss = val_running_loss/len(test_dataloader.dataset)
    val_accuracy = 100. * val_running_correct/len(test_dataloader.dataset)
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}')
    
    return val_loss, val_accuracy

# training function
def fit(model, train_dataloader, criterion, optimizer):
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    with tqdm(train_dataloader, unit="batch") as tepoch:
        for data in tepoch:
            data, target = data['image'].to(device, dtype=torch.float), data['label'].to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            train_running_loss += loss.item()
            _, preds = torch.max(output.data, 1)
            train_running_correct += (preds == target).sum().item()
            loss.backward()
            optimizer.step()
        train_loss = train_running_loss/len(train_dataloader.dataset)
        train_accuracy = 100. * train_running_correct/len(train_dataloader.dataset)
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}')
    
    return train_loss, train_accuracy

def run_training(model, savepath, epochs, train_dataloader, val_dataloader,
                criterion, optimizer):
    best_val_loss = np.inf
    train_loss , train_accuracy = [], []
    val_loss , val_accuracy = [], []
    start = time.time()
    epoch_number = 1
    for epoch in range(epochs):
        train_epoch_loss, train_epoch_accuracy = fit(model, train_dataloader, criterion, optimizer)
        val_epoch_loss, val_epoch_accuracy = validate(model, val_dataloader, criterion)
        if val_epoch_loss < best_val_loss:
            torch.save(model, savepath)
            best_val_loss = val_epoch_loss

        train_loss.append(train_epoch_loss)
        train_accuracy.append(train_epoch_accuracy)
        val_loss.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)
        print('Finished Epoch:',epoch_number)
        epoch_number += 1
    end = time.time()
    print((end-start)/60, 'minutes')