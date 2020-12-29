import torch
from tqdm import tqdm


def validate(model, val_loader, criterion):
    print('Validating')
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    dataset_length = len(val_loader.dataset)
    with torch.no_grad():
        for i, data in tqdm(enumerate(val_loader), total=int(dataset_length / val_loader.batch_size)):
            data, target = data[0].cuda(), data[1].cuda()
            outputs = model(data)
            y = torch.zeros(list(outputs.size())[0], 2)
            y[range(y.shape[0]), target] = 1
            y = y.cuda()
            loss = criterion(outputs, y)
            val_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            val_running_correct += (preds == target).sum().item()

        val_loss = val_running_loss / dataset_length
        val_accuracy = 100. * val_running_correct / dataset_length
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}')

        return val_loss, val_accuracy


def fit(model, train_loader, optimizer, criterion):
    print('Training')
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    dataset_length = len(train_loader.dataset)
    for i, data in tqdm(enumerate(train_loader), total=int(dataset_length / train_loader.batch_size)):
        if i == 0:
            print(dataset_length)
        data, target = data[0].cuda(), data[1].cuda()
        optimizer.zero_grad()
        outputs = model(data)
        y = torch.zeros(list(outputs.size())[0], 2)
        y[range(y.shape[0]), target] = 1
        y = y.cuda()
        loss = criterion(outputs, y)
        train_running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()

    train_loss = train_running_loss / dataset_length
    train_accuracy = 100. * train_running_correct / dataset_length

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}")

    return train_loss, train_accuracy
