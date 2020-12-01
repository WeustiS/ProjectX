import csv
import numpy as np
import torch
import os

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def import_dataset(data_path='option_call_dataset.txt'):
    assert os.path.isfile(data_path), "Data file does not exist"
    return np.loadtxt(data_path, delimiter=',')
    
def set_seed(seed, device=None):
    import numpy as np
    import torch
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        
def profit_hist(daily_profits, strategy='maximum average', battery='14kWh'):
    # Graphing
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.hist(daily_profits, bins=50, weights=np.ones(len(daily_profits)) / len(daily_profits))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.title("Daily profit for maximum average %s with %s battery" % (strategy, battery))
    plt.xlabel("Daily profit ($)")
    plt.ylabel("Number of days (%)")
    plt.legend(['Mean: %.2f, STD: %.2f' % (daily_profits.mean(), daily_profits.std())])
    plt.grid('minor')
    plt.xlim([-.5, 1.5])
    plt.show()
    
def plot_errs(err, var, model_name='Persistence'):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.errorbar([5 * i for i in range(1, 13)], 100*err, yerr=100*var)
    plt.title("%s Performance" % model_name)
    plt.grid(which='both')
    plt.ylim([0,65])
    plt.xlabel("Forecast Time (m)")
    plt.ylabel("Absolute Windspeed Percentage Error (%)")
    plt.show()