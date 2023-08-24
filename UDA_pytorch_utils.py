"""
Helper code for Carnegie Mellon University's Unstructured Data Analytics course
(95-865) that is targeted toward students in the Master of Information Systems
Management program.
Author: George H. Chen (georgechen [at symbol] cmu.edu)
Last update: Dec 10, 2022
I wrote this code to help make introducing neural nets & deep learning in
PyTorch relatively easy: the demos focus on the high-level ideas rather than
quickly getting into lower-level details which are instead hidden away in this
file, which I only talk about in the final lecture, time-permitting. In
particular, the course demos do not require students to know about which device
tensors are on, how batching is done, how minibatch gradient descent is carried
out, etc.
Right now, the code has only been tested using categorical cross entropy loss.
For students interested in better understanding what is happening in this file,
I would suggest first understanding the code logic that does not involve RNNs.
The parts of the code involving RNNs is quite a bit more involved. The added
complexity largely has to do with the time series data within a batch possibly
being of different lengths. To still store these different times series in a
single tensor, typically what is done is that each time series in the batch is
padded to be of the same length (e.g., they are all padded to be as long as
whichever time series in the batch has the most time steps). We separately also
keep track of how long the different time series are within the batch. There's
an alternative "packed" representation used by PyTorch; we convert between the
padded and packed versions in the current version of the code.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm


def UDA_pytorch_classifier_fit(model, optimizer, loss,
                               proper_train_dataset, val_dataset,
                               num_epochs, batch_size, device=None,
                               rnn=False,
                               save_epoch_checkpoint_prefix=None):
    """
    Trains a neural net classifier `model` using an `optimizer` such as Adam or
    stochastic gradient descent. We specifically minimize the given `loss`
    using the data given by `proper_train_dataset` using the number of epochs
    given by `num_epochs` and a batch size given by `batch_size`.
    Accuracies on the (proper) training data (`proper_train_dataset`) and
    validation data (`val_dataset`) are computed at the end of each epoch;
    `val_dataset` can be set to None if you don't want to use a validation set.
    The function outputs the training and validation accuracies.
    You can manually set which device (CPU or GPU) to use with the optional
    `device` argument (e.g., setting `device=torch.device('cpu')` or
    `device=torch.device('cuda')`). By default, the code tries to use a GPU if
    it is available.
    The boolean argument `rnn` says whether we are looking at time series
    data (set this True for working with recurrent neural nets).
    
    Lastly, if `save_epoch_checkpoint_prefix` is a string prefix, then each
    epoch's model is saved to a filename with format
    '<save_epoch_checkpoint_prefix>_epoch<epoch number>.pt'.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if loss._get_name() != 'CrossEntropyLoss':
        raise Exception('Unsupported loss: ' + loss._get_name())

    # PyTorch uses DataLoader to load data in batches
    if not rnn:
        proper_train_loader = \
            torch.utils.data.DataLoader(dataset=proper_train_dataset,
                                        batch_size=batch_size,
                                        shuffle=True)
        if val_dataset is not None:
            val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                     batch_size=batch_size,
                                                     shuffle=False)
    else:
        # variable-length time series data are handled in a different manner;
        # we can still use DataLoader but use a custom `collate_fn` for how
        # processing each batch
        proper_train_loader = \
            torch.utils.data.DataLoader(
                    dataset=proper_train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=UDA_collate_variable_length_batch)
        if val_dataset is not None:
            val_loader = torch.utils.data.DataLoader(
                    dataset=val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=UDA_collate_variable_length_batch)

    train_accuracies = np.zeros(num_epochs)
    val_accuracies = np.zeros(num_epochs)

    for epoch_idx in tqdm(range(num_epochs)):
        # go through training data
        num_training_examples_so_far = 0
        for batch_data in proper_train_loader:
            if not rnn:
                batch_inputs, batch_labels = batch_data

                # make sure the data are stored on the right device
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.to(device)

                # make predictions for current batch and compute loss
                batch_outputs = model(batch_inputs)
            else:
                batch_padded_text_encodings, batch_lengths, batch_labels \
                    = batch_data

                batch_padded_text_encodings \
                    = batch_padded_text_encodings.to(device)
                batch_lengths = batch_lengths.to(device)
                batch_labels = batch_labels.to(device)

                # make predictions for current batch and compute loss
                batch_outputs = model(batch_padded_text_encodings,
                                      batch_lengths)

            batch_loss = loss(batch_outputs, batch_labels)

            # update model parameters
            optimizer.zero_grad()  # reset which direction optimizer is going
            batch_loss.backward()  # compute new direction optimizer should go
            optimizer.step()  # move the optimizer

        # compute proper training and validation set raw accuracies
        model.eval()  # turn on evaluation mode
        train_accuracy = \
            UDA_compute_accuracy(
                UDA_pytorch_classifier_predict(model,
                                               proper_train_dataset,
                                               device=device,
                                               batch_size=batch_size,
                                               rnn=rnn,
                                               inputs_have_labels=True),
                [label for input, label in proper_train_dataset])
        print('  Train accuracy: %.4f' % train_accuracy, flush=True)
        train_accuracies[epoch_idx] = train_accuracy

        if val_dataset is not None:
            val_accuracy = \
                UDA_compute_accuracy(
                    UDA_pytorch_classifier_predict(model,
                                                   val_dataset,
                                                   device=device,
                                                   batch_size=batch_size,
                                                   rnn=rnn,
                                                   inputs_have_labels=True),
                    [label for input, label in val_dataset])
            print('  Validation accuracy: %.4f' % val_accuracy, flush=True)
            val_accuracies[epoch_idx] = val_accuracy
        model.train()  # turn off evaluation mode

        if save_epoch_checkpoint_prefix is not None:
            torch.save(model.state_dict(),
                       '%s_epoch%d.pt'
                       % (save_epoch_checkpoint_prefix, epoch_idx + 1))

    return train_accuracies, val_accuracies


def UDA_pytorch_model_transform(model, inputs, device=None, batch_size=128,
                                rnn=False, inputs_have_labels=False):
    """
    Given a neural net `model`, evaluate the model given `inputs`, which should
    *not* be already batched. This helper function automatically batches the
    data, feeds each batch through the neural net, and then unbatches the
    outputs. The outputs are stored as a PyTorch tensor.
    You can manually set which device (CPU or GPU) to use with the optional
    `device` argument (e.g., setting `device=torch.device('cpu')` or
    `device=torch.device('cuda')`). By default, the code tries to use a GPU if
    it is available.
    You can also manually set `batch_size`; this is less critical than in
    training since we are, at this point, just evaluating the model without
    updating its parameters.
    The boolean argument `rnn` says whether we are looking at time
    series data (set this True for working with recurrent neural nets).
    Lastly, the boolean argument `inputs_have_labels` says whether `inputs`
    contain labels or not. Basically if labels are included, we will ignore
    them as they are not needed for applying the model to the raw inputs.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # batch the inputs
    if not rnn:
        data_loader = torch.utils.data.DataLoader(dataset=inputs,
                                                  batch_size=batch_size,
                                                  shuffle=False)
    else:
        if inputs_have_labels:
            data_loader = torch.utils.data.DataLoader(
                dataset=inputs, batch_size=batch_size, shuffle=False,
                collate_fn=UDA_collate_variable_length_batch)
        else:
            data_loader = torch.utils.data.DataLoader(
                dataset=inputs, batch_size=batch_size, shuffle=False,
                collate_fn=UDA_collate_variable_length_batch_no_labels)

    outputs = []
    with torch.no_grad():
        for batch_inputs in data_loader:
            if inputs_have_labels:  # ignore labels
                if not rnn:
                    # in this case, `batch_inputs` should be a tuple of the
                    # format (inputs, labels) for which we only take the
                    # content in index 0
                    batch_inputs = batch_inputs[0]
                else:
                    batch_inputs = batch_inputs[:-1]

            if not rnn:
                batch_inputs = batch_inputs.to(device)
                batch_outputs = model(batch_inputs)
            else:
                batch_padded_text_encodings, batch_lengths = batch_inputs

                batch_padded_text_encodings \
                    = batch_padded_text_encodings.to(device)
                batch_lengths = batch_lengths.to(device)

                batch_outputs = model(batch_padded_text_encodings,
                                      batch_lengths)
            outputs.append(batch_outputs)

    return torch.cat(outputs, 0)


def UDA_pytorch_classifier_predict(model, inputs, device=None, batch_size=128,
                                   rnn=False, inputs_have_labels=False):
    """
    See the documentation for `UDA_pytorch_model_transform`. This function has
    the same arguments but the output is the final predicted class per data
    point rather than the raw neural net output per data point.
    """
    outputs = UDA_pytorch_model_transform(
            model, inputs, device=device, batch_size=batch_size, rnn=rnn,
            inputs_have_labels=inputs_have_labels)

    with torch.no_grad():
        return outputs.argmax(axis=1).view(-1)


def UDA_plot_train_val_accuracy_vs_epoch(train_accuracies, val_accuracies):
    """
    Helper function for plotting (proper) training and validation accuracies
    across epochs; `train_accuracies` and `val_accuracies` should be the same
    length, which should equal the number of epochs.
    """
    ax = plt.figure().gca()
    num_epochs = len(train_accuracies)
    plt.plot(np.arange(1, num_epochs + 1), train_accuracies, '-o',
             label='Training')
    plt.plot(np.arange(1, num_epochs + 1), val_accuracies, '-+',
             label='Validation')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def UDA_compute_accuracy(labels1, labels2):
    """
    Computes the raw accuracy of two label sequences `labels1` and `labels2`
    agreeing. This helper function coerces both label sequences to be on the
    CPU, flattened, and stored as 1D NumPy arrays before computing the average
    agreement.
    """
    if type(labels1) == torch.Tensor:
        labels1 = labels1.detach().view(-1).cpu().numpy()
    elif type(labels1) != np.ndarray:
        labels1 = np.array(labels1).flatten()
    else:
        labels1 = labels1.flatten()

    if type(labels2) == torch.Tensor:
        labels2 = labels2.detach().view(-1).cpu().numpy()
    elif type(labels2) != np.ndarray:
        labels2 = np.array(labels2).flatten()
    else:
        labels2 = labels2.flatten()

    return np.mean(labels1 == labels2)


def UDA_collate_variable_length_batch(batch):
    """
    Custom collating code for use with the RNN demo; this version assumes that
    labels are available.
    """
    text_encodings = []
    lengths = []
    labels = []

    for text_encoding, label in batch:
        if type(text_encoding) is torch.Tensor:
            text_encodings.append(text_encoding)
        else:
            text_encodings.append(torch.tensor(text_encoding))
        lengths.append(len(text_encoding))
        labels.append(label)

    padded_text_encodings = nn.utils.rnn.pad_sequence(text_encodings)
    lengths = torch.tensor(lengths, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_text_encodings, lengths, labels


def UDA_collate_variable_length_batch_no_labels(batch):
    """
    Custom collating code for use with the RNN demo; this version assumes that
    labels are *not* available.
    """
    text_encodings = []
    lengths = []

    for text_encoding in batch:
        if type(text_encoding) is torch.Tensor:
            text_encodings.append(text_encoding)
        else:
            text_encodings.append(torch.tensor(text_encoding))
        lengths.append(len(text_encoding))

    padded_text_encodings = nn.utils.rnn.pad_sequence(text_encodings)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return padded_text_encodings, lengths


def UDA_get_rnn_last_time_step_outputs(padded_inputs, lengths, rnn):
    """
    Applies an RNN (such as an LSTM) to a batch of time series and extracts the
    last output per time series.
    The batch of time series are represented in a padded format `padded_inputs`
    where the actual time series lengths are in `lengths`. The RNN nn.Module to
    be applied is `rnn`.
    """
    # some helper code that we use from PyTorch requires that the batch's time
    # series are sorted in decreasing order of length; we will apply this
    # sorting now and then undo the sorting at the end of this function
    sorted_lengths, sort_indices = torch.sort(lengths, descending=True)
    sorted_padded_inputs = padded_inputs[:, sort_indices, :]

    # there's a weird issue where lengths must be converted to be on the CPU
    # for `pack_padded_sequence` even if all the data resides on the GPU:
    # https://github.com/pytorch/pytorch/issues/43227
    packed_rnn_sequence = nn.utils.rnn.pack_padded_sequence(
        sorted_padded_inputs, sorted_lengths.cpu())

    # apply the RNN to the packed RNN sequence to get a packed RNN output
    packed_rnn_output, _ = rnn(packed_rnn_sequence)

    # undo the packing of the output by converting it into a padded time series
    padded_rnn_output, _ = nn.utils.rnn.pad_packed_sequence(
        packed_rnn_output, total_length=sorted_lengths[0].item())

    # extract only the final time step per time series (final time step is
    # the length minus 1)
    rnn_last_time_step_outputs = \
        padded_rnn_output[[length - 1 for length in sorted_lengths],
                          [idx for idx in range(padded_inputs.shape[1])], :]

    # undo the sorting applied at the start of the function
    return rnn_last_time_step_outputs[sort_indices.argsort(0)]