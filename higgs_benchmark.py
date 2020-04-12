import time
from typing import List

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score

from fast_tensor_data_loader import FastTensorDataLoader

# data params
ROW_LIMIT = None # for quicker testing
NUM_TEST_ROWS = 500000
LABEL_COLUMN = 0
FEATURE_COLUMNS = list(range(1, 22)) # low-level features only as per http://archive.ics.uci.edu/ml/datasets/HIGGS
FILE_NAME = '/home/ubuntu/HIGGS.csv.gz'
GPU = True

# hyperparams
BATCH_SIZE = 16384
NUM_EPOCHS = 10

def load_data(file_name: str, test_rows: int, feature_columns: List[int], label_column: int, row_limit: int):
    """ Load data from disk into train and test tensors """
    # load csv file
    data = pd.read_csv(file_name, header=None, dtype='float32', nrows=row_limit)

    features = torch.from_numpy(data.loc[:, feature_columns].reset_index(drop=True).values)
    labels = torch.from_numpy(data.loc[:, label_column].reset_index(drop=True).values)

    train_x = features[:-test_rows]
    train_y = labels[:-test_rows]

    test_x = features[-test_rows:]
    test_y = labels[-test_rows:]

    return train_x, train_y, test_x, test_y


# load data
train_x, train_y, test_x, test_y = load_data(file_name=FILE_NAME, test_rows=NUM_TEST_ROWS,
            feature_columns=FEATURE_COLUMNS, label_column=LABEL_COLUMN, row_limit=ROW_LIMIT)

def create_model(gpu: bool=True):
    """
    Create a PyTorch neural net of depth 4. Architecture based on
    https://static-content.springer.com/esm/art%3A10.1038%2Fncomms5308/MediaObjects/41467_2014_BFncomms5308_MOESM1199_ESM.pdf. 
    """
    dropout_rate = 0.5
    hidden_units = 300
    model = nn.Sequential(
        nn.Linear(len(FEATURE_COLUMNS), hidden_units),
        nn.Tanh(),
        nn.Dropout(p=dropout_rate), 
        nn.Linear(hidden_units, hidden_units),
        nn.Tanh(),
        nn.Linear(hidden_units, hidden_units),
        nn.Tanh(),
        nn.Linear(hidden_units, hidden_units),
        nn.Tanh(),
        nn.Linear(hidden_units, 1),
        nn.Sigmoid()
        )
    if gpu: # move model onto the GPU
        model = model.cuda()
    return model

def train_for_n_epochs(model: nn.Module, train_dataloader: DataLoader, test_x: torch.Tensor, test_y: torch.Tensor, epochs: int, name: str, log_every_n_steps=100, eval_every_n_steps=100, gpu: bool=True):
    """ Train a pytorch model for a set number of epochs, using a given dataloader. """
    optimizer = torch.optim.Adam(params=model.parameters())
    loss_fn = nn.BCELoss(reduction='mean')

    # log to tensorboard
    writer = SummaryWriter(comment=name)

    global_step = 0

    if gpu: # move test set on to GPU
        test_x = test_x.cuda()
        test_y = test_y.cuda()

    for epoch in range(epochs):
        for x_batch, y_batch in train_dataloader:
            if gpu: # move batches from RAM into GPU memory
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()
            y_pred = model(x_batch)

             # zero the parameter gradients
            optimizer.zero_grad()

            loss = loss_fn(y_pred.squeeze(), y_batch)

            if global_step % log_every_n_steps == log_every_n_steps - 1:
                    writer.add_scalar('Loss/train', loss, global_step)
                    writer.add_scalar('Epoch', epoch, global_step)
                    roc_auc = roc_auc_score(y_batch.cpu().numpy(), y_pred.cpu().detach().numpy())
                    writer.add_scalar('ROC_AUC/train', roc_auc, global_step)

            loss.backward()
            optimizer.step()

            if global_step % eval_every_n_steps == eval_every_n_steps - 1:
                    test_y_pred = model(test_x)
                    test_loss = loss_fn(test_y_pred.squeeze(), test_y)
                    writer.add_scalar('Loss/test', test_loss, global_step)
                    test_roc_auc = roc_auc_score(test_y.numpy(), test_y_pred.detach().numpy())
                    writer.add_scalar('ROC_AUC/test', test_roc_auc, global_step)

            global_step += 1
        print(f'Epoch {epoch} done. ')


data_set = TensorDataset(train_x, train_y)
default_train_batches = DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=False)
fast_train_batches = FastTensorDataLoader(train_x, train_y, batch_size=BATCH_SIZE, shuffle=False)

# standard dataloader benchmark
model = create_model(gpu=GPU)
start = time.perf_counter()

train_for_n_epochs(model=model, train_dataloader=default_train_batches, test_x=test_x, test_y=test_y, epochs=NUM_EPOCHS, name='default_data_loader', gpu=GPU)

default_elapsed_seconds = time.perf_counter() - start

# improved dataloader benchmark
model = create_model(gpu=GPU)
start = time.perf_counter()

train_for_n_epochs(model=model, train_dataloader=fast_train_batches, test_x=test_x, test_y=test_y, epochs=NUM_EPOCHS, name='custom_data_loader', gpu=GPU)

fast_elapsed_seconds = time.perf_counter() - start

print(f'Standard dataloader: {default_elapsed_seconds/NUM_EPOCHS:.2f}s/epoch.')
print(f'Custom dataloader: {fast_elapsed_seconds/NUM_EPOCHS:.2f}s/epoch.')
