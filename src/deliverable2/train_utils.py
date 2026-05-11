import torch


def train(model, train_loader, optimizer, loss_fn, device):
    model.train()
    for data, target in train_loader:
        # Move batch to device
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        logits = model(data)
        # Multiclass classification: logits must be [batch_size, num_classes]
        loss = loss_fn(logits, target)
        loss.backward()
        optimizer.step()


def validation(model, val_loader, loss_fn, device) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            total += loss_fn(logits, target).item()
    return total / len(val_loader)


def predict(model, test_loader, device, scaler_y):
    model.eval()
    preds = []
    reals = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            pred_class = logits.argmax(dim=1)
            preds.append(pred_class.cpu())
            reals.append(y_batch.cpu())

    y_pred = torch.cat(preds)
    y_true = torch.cat(reals)

    # If target encoder is a LabelEncoder, return original class labels.
    if hasattr(scaler_y, "classes_"):
        y_pred_original = scaler_y.inverse_transform(y_pred.numpy())
        y_true_original = scaler_y.inverse_transform(y_true.numpy())
        return y_pred_original, y_true_original

    return y_pred.numpy(), y_true.numpy()

def compute_accuracy(model, data_loader, device):

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)

            # Get predictions
            predictions = torch.argmax(output, dim=1)

            # Count correct predictions
            correct += (predictions == target).sum().item()
            total += target.size(0)

    accuracy = correct / total if total > 0 else 0.0
    return accuracy