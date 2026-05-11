from sklearn.preprocessing import StandardScaler, LabelEncoder
from pandas.api.types import is_numeric_dtype
import torch
from torch.utils.data import TensorDataset, DataLoader


def prepare_data(X_train, X_test, X_val , y_train, y_test , y_val) :

    # Standardize the features using StandardScaler from the train Data
    scaler_x = StandardScaler()
    X_train = scaler_x.fit_transform(X_train)
    X_test = scaler_x.transform(X_test)
    X_val = scaler_x.transform(X_val)


    # Categorical target: encode labels to integers. We do NOT apply StandardScaler to labels.
    le = LabelEncoder()
    y_train_proc = le.fit_transform(y_train.values)
    y_test_proc = le.transform(y_test.values)
    y_val_proc = le.transform(y_val.values)
    scaler_y = le

    # Convert to tensors Dataset so we can parse to the DataLoader
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)

    # Convert processed targets to tensors. Use float for regression and long (int64) for classification targets.
    y_train = torch.tensor(y_train_proc, dtype=torch.long)
    y_test = torch.tensor(y_test_proc, dtype=torch.long)
    y_val = torch.tensor(y_val_proc, dtype=torch.long)


    # Once the train/test splits are converted to PyTorch tensors,wrap them in TensorDataset so each sample is returned as (features, target).
    train_td = TensorDataset(X_train, y_train)
    test_td = TensorDataset(X_test, y_test)
    val_td = TensorDataset(X_val, y_val)

    # DataLoader groups samples into mini-batches.
    # Example with batch_size=64:
    #   X batch shape -> [64, 8]
    #   y batch shape -> [64]
    # The model will output [64, 1] (one value per sample, 64 samples in the batch).
    train_loader = DataLoader(train_td, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_td, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_td, batch_size=64, shuffle=False)

    return train_loader,  val_loader, test_loader ,scaler_x, scaler_y