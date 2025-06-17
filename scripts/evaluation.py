import numpy as np
import torch
from matplotlib import pyplot as plt
import seaborn as sns

def get_predictions(model, dataloader):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        predictions = []
        targets = []

        for features, labels in dataloader:
            features = features.to(device)
            prediction = model(features)
            predictions.append(prediction.cpu().numpy().reshape(-1))
            targets.append(labels.cpu().numpy().reshape(-1))

    return predictions, targets

def plot_predictions(model, test_loader, file_path=None, directory=None):
    def name_plot(plot_type):
        name = file_path.split('/')[-1].split('.')[0] + '_' + plot_type
        plt.xlabel('Data Points')
        plt.ylabel('SOC')
        plt.title(name)
        return name

    def save_plot(name):
        if directory:
            file_path = directory / name
            plt.savefig(file_path)

    predictions, targets = get_predictions(model, test_loader)
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    # plot only the 300th prediction (time needed for lstm to adjust and find SOC value)
    predictions = predictions[::299]
    targets = targets[::299]

    print(predictions[0].shape)

    print(f'File : {file_path}')
    print(f'Predictions: {predictions[:20]}')
    print(f'Targets: {targets[:20]}')

    print(predictions.shape)

    sns.lineplot(x=range(len(targets)), y=targets, label='Actual', color='red')
    sns.lineplot(x=range(len(predictions)), y=predictions, label='Predicted', color='blue')
    name = name_plot('lineplot')
    save_plot(name)
    plt.show()

    sns.scatterplot(x=range(len(targets)), y=targets, label='Actual', color='red')
    sns.scatterplot(x=range(len(predictions)), y=predictions, label='Predicted', color='blue')
    name = name_plot('scatterplot')
    save_plot(name)
    plt.show()

'''

# plot results with given title and save file




        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            predictions.append(outputs.cpu().numpy().reshape(-1))
            targets.append(batch_y.numpy().reshape(-1))

    return predictions, targets
    
plot data extraction for cleaned features
    
    plt.xlabel('Data Points')
    plt.ylabel('SOC')
    plt_name = name + ' Lineplot'
    plt.title(plt_name)

    if path:
        filename = '_'.join(plt_name.lower().split(' '))
        file_path = path / filename
        plt.savefig(file_path)

    plt.show()

    sns.scatterplot(x=range(len(targets)), y=targets, label='Actual', color='red')
    sns.scatterplot(x=range(len(predictions)), y=predictions, label='Predicted', color='blue')
    plt.xlabel('Data Points')
    plt.ylabel('SOC')
    plt_name = name + ' Scatterplot'
    plt.title(plt_name)

    if path:
        filename = '_'.join(plt_name.lower().split(' '))
        file_path = path / filename
        plt.savefig(file_path)

    plt.show()

'''
