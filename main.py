import os
import statistics
from functools import partial

import torch
import pyreadstat
import numpy as np
import pandas as pd
import torch.nn as nn
import networkx as nx
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torchmetrics.classification import MulticlassAUROC

from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import tempfile
from ray import train, tune
from ray.train import Checkpoint

# Logistic Regression Training Hyperparameters
HYPERPARAMETERS = {
    'lr': tune.loguniform(1e-4, 1e-1),
    'weight_decay': tune.loguniform(1e-4, 1e-1),
    'num_epochs': tune.choice([2, 16, 64, 512, 1024, 4096, 8192]),
}

def get_COPDGeneSOMASCAN_dataset():
    COPDGene_SOMA_13_dataset = pd.read_csv('/home/shussein/NetCO/data/SOMASCAN13/COPDGene_SOMASCAN13_subjects.csv')
    return COPDGene_SOMA_13_dataset


def get_datasets():
    return get_COPDGeneSOMASCAN_dataset()


def get_graph_data():
    ppi_graph_adj = pd.read_csv('/home/shussein/NetCO/data/PPI_Yong/ppi_graph_1183_mRNA_updated_root_2656sub.csv', delimiter='\t').to_numpy()
    ppi_graph_nx = nx.from_numpy_array(ppi_graph_adj)
    return ppi_graph_nx


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


class GraphRegularizedLoss(nn.Module):
    def __init__(self, laplacian_matrix, lambda_reg):
        super(GraphRegularizedLoss, self).__init__()
        # self.adjacency_matrix = adjacency_matrix
        self.laplacian_matrix = laplacian_matrix
        self.lambda_reg = lambda_reg

    def forward(self, weights_matrix):
        loss_reg = 0
        for weight_vector in weights_matrix:
            mult = self.laplacian_matrix.dot(weight_vector.detach().abs())
            l = np.sum(mult)
            loss_reg += l

        return self.lambda_reg * loss_reg
def train_n(hyperparameters):
    hyperparams_tuning = False
    dataset = get_datasets()
    graph_data = get_graph_data()

    # TODO: We assume that the phenotype column is named 'finalgold_visit'
    X = dataset.loc[:, dataset.columns != 'finalgold_visit']
    Y = dataset['finalgold_visit']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to PyTorch Tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train.values)

    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.LongTensor(y_test.values)

    input_dim = X_train_tensor.shape[1]
    output_dim = 3  # Number of Classes # TODO: Make it dynamic

    lr_model = LogisticRegression(input_dim=input_dim, output_dim=output_dim)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # TODO: Need to review this implementaion
    graph_regularized_loss = GraphRegularizedLoss(nx.normalized_laplacian_matrix(graph_data), lambda_reg=0.01)
    # TODO: Try other optimizers
    optimizer = optim.SGD(lr_model.parameters(), lr=hyperparameters['lr'], weight_decay=hyperparameters['weight_decay'])

    # Training loop
    for epoch in range(hyperparameters['num_epochs']):
        optimizer.zero_grad()
        outputs = lr_model(X_train_tensor)

        weights = lr_model.linear.weight
        loss = criterion(outputs, y_train_tensor)

        graph_regularized_loss1 = graph_regularized_loss(weights)
        total_loss = loss + graph_regularized_loss1

        total_loss.backward()
        optimizer.step()

        # Print loss for monitoring


        # print("*****************GRADIENTS********************")
        # print("Gradients for the Model!!!! ")
        # # Track gradients (gnn_model) and model parameters in TensorBoard
        # for name, param in model.named_parameters():
        #     print(f'{name}.grad', param.grad, epoch)

        if hyperparams_tuning:
            metrics = {"loss": loss.item()}
            with tempfile.TemporaryDirectory() as tempdir:
                torch.save(
                    {"epoch": epoch, "model_state": lr_model.state_dict()},
                    os.path.join(tempdir, "checkpoint.pt"),
                )
                train.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tempdir))

        with torch.no_grad():
            lr_model.eval()
            outputs = lr_model(X_test_tensor)  # .flatten()
            _, predicted = torch.max(outputs, 1)

            accuracy = torch.sum(predicted == y_test_tensor).item() / len(y_test)
            mse = nn.MSELoss()(predicted, y_test_tensor.float())

            confusion_mtrx = confusion_matrix(y_test_tensor, predicted)
            # mc_auroc = MulticlassAUROC(num_classes=3, average='macro', thresholds=None)
            # print(f'Classification Report: {classification_report(y_test_tensor, predicted)}')
            # print(f'ROC AUC Score: {mc_auroc(outputs, y_test_tensor)}')
            # print(f'Confusion Matrix: {confusion_mtrx}')
            if hyperparams_tuning:
                metrics["accuracy"] = accuracy
                with tempfile.TemporaryDirectory() as tempdir:
                    train.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tempdir))
    # print("Parameters %s" % lr_model.state_dict())
    print(f"Epoch [{epoch + 1}/{epoch}],\tTask Loss: {loss.item()},\tGraph Loss: {graph_regularized_loss1.item()},\tTotal: {total_loss.item()}")
    print(f"Epoch [{epoch + 1}/{epoch}],\tLoss: {loss.item()},\tAccuracy: {accuracy},\tMSE: {mse}")
    return accuracy, mse


def main():
    num_trials = 1000
    test_accuracies = []
    test_mses = []
    for i in range(num_trials):
        print(f"Trial {i}")
        hyperparams_tuning = False
        if hyperparams_tuning:
            reporter = CLIReporter(metric_columns=["loss"])
            GRACE_PERIOD = 10
            # Scheduler to stop bad performing trials
            scheduler = ASHAScheduler(
                metric="loss",
                mode="min",
                grace_period=GRACE_PERIOD,
                reduction_factor=2)

            # TODO: Remove this higher limit on the number of trials
            NUM_SAMPLES = 5000
            result = tune.run(partial(train_n),
                              resources_per_trial={"cpu": 8, "gpu": 0},
                              hyperparameters=HYPERPARAMETERS, num_samples=NUM_SAMPLES, scheduler=scheduler,
                              # local_dir='outputs/raytune_results_%s' % experiment_name,
                              name='PredictionNNGlobalGraph',
                              keep_checkpoints_num=1,
                              checkpoint_score_attr='min-loss',
                              progress_reporter=reporter
                              )
            best_trial = result.get_best_trial("loss", "min", "last")
            print("Best trial config: {}".format(best_trial.config))
            print("Best trial final test loss: {}".format(best_trial.last_result["loss"]))
        else:
            test_accuracy, test_mse = train_n(hyperparameters={
                'lr': 0.030254,
                'weight_decay': 0.000100,
                'num_epochs': 1000,
            })

            test_accuracies.append(test_accuracy)
            test_mses.append(test_mse)
    print("Best Accuracy is {}".format(max(test_accuracies)))
    print("Average Accuracy is {}".format(sum(test_accuracies)/len(test_accuracies)))
    print("Standard Deviation is {}".format(statistics.stdev(test_accuracies)))
    print("----------------")
    print("Average MSE is {}".format(sum(test_mses)/len(test_mses)))
    # print("Standard Deviation is {}".format(statistics.stdev(test_mses)))


if __name__ == "__main__":
    main()
