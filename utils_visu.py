import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_gp(ax, mean, std, title, min_x=0, max_x=1):
    X_plot = np.linspace(min_x, max_x, 1000).reshape(-1, 1)
    ax.plot(X_plot, mean, color="blue", label="Surrogate model")
    ax.fill_between(X_plot[:, 0], mean - 2 * std, mean + 2 * std, color="blue", alpha=0.2)
    ax.set_title(title)
    ax.legend()
    
    
def plot_biais_variance(data, true_loss, x="time", savefig = None, palette=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,5))
    sns.lineplot(data=data, x=x, y="scores", hue="series", errorbar="sd", ax=ax1, palette=palette)
    ax1.hlines(true_loss, xmin=1, xmax= max(data[x]), colors='red', label = 'True loss')
    ax1.set_ylabel('Bias')
    df_var = (data.groupby([x, 'series'])['scores'].agg('var').reset_index())
    sns.lineplot(data=df_var, x=x, y="scores",  hue="series", ax=ax2, palette=palette)
    ax2.set_yscale('log')
    ax2.set_ylabel('Variance')
    if savefig:
        plt.savefig(savefig+'_'+x+'.pdf', format = 'pdf')
    plt.show()
    
    
def plot_biais(data, true_loss, x="time", savefig=None, palette=None):
    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    sns.lineplot(data=data, x=x, y="scores", hue="series", errorbar="sd", ax=ax, palette=palette)
    ax.hlines(true_loss, xmin=1, xmax=max(data[x]), colors='red', label='True loss')
    ax.set_ylabel('Bias')
    
    if x=='time':
        ax.set_xlabel('Number of Seen Points in the Stream')
    elif x == 'nb_selected':
        ax.set_xlabel('Number of Acquired Test Points')

    if savefig:
        plt.savefig(savefig+'_'+x+"bias"+'.pdf', format='pdf')
    plt.show()

def plot_variance(data, true_loss, x="time", savefig=None, palette=None):
    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    df_var = (data.groupby([x, 'series'])['scores'].agg('var').reset_index())
    sns.lineplot(data=df_var, x=x, y="scores", hue="series", ax=ax, palette=palette)
    ax.set_yscale('log')
    ax.set_ylabel('Variance')
    
    if x=='time':
        ax.set_xlabel('Number of Seen Points in the Stream')
    elif x == 'nb_selected':
        ax.set_xlabel('Number of Acquired Test Points')

    if savefig:
        plt.savefig(savefig+'_'+x+"variance"+'.pdf', format='pdf')
    plt.show()
