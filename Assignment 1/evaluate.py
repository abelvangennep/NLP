"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch
import utils
import model.net as net
from model.data_loader import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/preprocessed', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")


def evaluate(model, loss_fn, data_iterator, metrics, num_steps):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        data_iterator: (generator) a generator that generates batches of data and labels
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    for _ in range(num_steps):
        # fetch the next evaluation batch
        data_batch, labels_batch = next(data_iterator)
        
        # compute model output
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean


####
# MY ADJUSTMENT: same as evaluate, but also outputs the labels
def evaluate_and_output(model, loss_fn, data_iterator, metrics, num_steps, id2word, tags, outfile):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        data_iterator: (generator) a generator that generates batches of data and labels
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    with open(outfile, "w", encoding="utf-8") as f:
        for _ in range(num_steps):
            # fetch the next evaluation batch
            data_batch, labels_batch = next(data_iterator)

            # compute model output
            output_batch = model(data_batch)
            loss = loss_fn(output_batch, labels_batch)

            # extract data from torch Variable, move to cpu, convert to numpy arrays
            data_batch = data_batch.data.cpu().numpy()
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            # reshape labels to give a flat vector of length batch_size*seq_len
            seq_len = labels_batch.shape[1]
            labels = labels_batch.ravel()
            words = data_batch.ravel()

            # np.argmax gives us the class predicted for each token by the model
            outputs = np.argmax(output_batch, axis=1)

            index = 0
            for i in range(len(labels)):
                # -1 is the label for the padding items
                index += 1
                if not labels[i] == -1:
                    word = id2word[words[i]]
                    gold = tags[labels[i]]
                    prediction = tags[outputs[i]]
                    f.write("\t".join([word, gold, prediction]))
                    f.write("\n")
                if index == seq_len:
                    index = 0
                    f.write("----------\n")
            # compute all metrics on this batch
            summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                             for metric in metrics}
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean

if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)
        
    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # load data
    data_loader = DataLoader(args.data_dir, params)
    data = data_loader.load_data(['test'], args.data_dir)
    test_data = data['test']

    # specify the test set size
    params.test_size = test_data['size']
    test_data_iterator = data_loader.data_iterator(test_data, params)

    logging.info("- done.")

    # Define the model
    model = net.Net(params).cuda() if params.cuda else net.Net(params)
    
    loss_fn = net.loss_fn
    metrics = net.metrics
    
    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    num_steps = (params.test_size + 1) // params.batch_size
    # MY ADJUSTMENTS
    # reverse the vocab and tag dictionary to be able to map back from ids to words and tags
    id2word = {v: k for k, v in data_loader.vocab.items()}
    tags = {v: k for k, v in data_loader.tag_map.items()}
    outfile = args.model_dir + "/25epoch.tsv"
    test_metrics = evaluate_and_output(model, loss_fn, test_data_iterator, metrics, num_steps, id2word, tags, outfile)
    save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)

