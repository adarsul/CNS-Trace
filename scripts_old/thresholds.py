import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from scipy.integrate import quad

from model import *


random.seed(123)


def import_seqs(fasta):
    seqs = {}
    with open(fasta,'r') as file:
        for row in file:
            if row.startswith('>'):
                name = row.strip('>\n')
            else:
                seq = row
                seqs[name] = seq.strip('\n')
    return seqs


def shuffle(s:str):
    """Takes a string and returns a shuffled version of it

    Args:
        s (str): The given string

    Returns:
        shuffled_string (str): a shuffled string
    """
    # Convert the string to a list of characters
    char_list = list(s)
    # Shuffle the list
    random.shuffle(char_list)
    # Join the characters back into a string
    shuffled_string = ''.join(char_list)
    return shuffled_string

def threshold_plot(TF,pwm,x,y,true_pdf,bg_pdf,energies,shuffled_energies,quarters):
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Histograms
    sns.histplot(energies, stat='density', label='DAP-seq', ax=ax1, alpha = 0.4)
    sns.histplot(shuffled_energies, color='orange', stat='density', label='Shuffled', alpha=0.4, ax=ax1)

    # Fitted disitribution
    line, = ax1.plot(x, true_pdf, label = 'P(E)')
    line2, = ax1.plot(x, bg_pdf, label = 'Q(E)')
    ax1.fill_between(x, true_pdf, alpha=0.3, color=line.get_color())
    ax1.fill_between(x, bg_pdf, alpha=0.3, color=line2.get_color())

    # Log prob plot
    ax2 = ax1.twinx()  # Create a secondary y-axis

    ax2.plot(x,y, color = 'k', label = 'S(E)')
    for x_val in quarters:
        plt.axvline(x_val, color= 'k', linestyle = 'dashed')

    ax1.set_xlabel('E (Energy Score)')
    ax1.set_ylabel('Density')
    ax2.set_ylabel('S(E)')


    # legend

    # Collect handles and labels from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # Combine handles and labels, then create the legend
    handles = handles1 + handles2
    labels = labels1 + labels2
    ax1.legend(handles, labels, loc='upper right')
    plt.title(f"{TF} Energy distributions - N={len(shuffled_energies)}, {len(pwm['A'])}mer")

    return fig, ax1, ax2

def get_dap_and_shuffled_energies(dap_seqs, pwm, gc_dict):
    energies = np.zeros(len(dap_seqs))
    shuffled_energies = np.zeros(len(dap_seqs))

    # Get energies
    for i, sequence in enumerate(dap_seqs.values()):
        energies[i] = getBindingEnergy(sequence, pwm, gc_dict, species = 'Athaliana', energy_only = True)
        shuffled_energies[i] = getBindingEnergy(shuffle(sequence), pwm, gc_dict, species = 'Athaliana', energy_only = True)
    
    energies = energies[energies < np.percentile(energies, 95)]
    
    return energies, shuffled_energies

# def fit_distributions(dap_energies, shuffled_energies):
#     try:
#         # fit models
#         dap_lognorm = lognorm.fit(dap_energies)
#         shuf_lognorm = lognorm.fit(shuffled_energies)
#     except:
#         dap_energies = dap_energies[np.isfinite(dap_energies)]
#         shuffled_energies = shuffled_energies[np.isfinite(shuffled_energies)]
#         dap_lognorm = lognorm.fit(dap_energies)
#         shuf_lognorm = lognorm.fit(shuffled_energies)

#     return dap_lognorm, shuf_lognorm

def fit_distributions(dap_energies, shuffled_energies):
    try:
        # Ensure values are valid for fitting
        if not np.all(np.isfinite(dap_energies)):
            print("Invalid values in dap_energies:", dap_energies[~np.isfinite(dap_energies)])
        if not np.all(np.isfinite(shuffled_energies)):
            print("Invalid values in shuffled_energies:", shuffled_energies[~np.isfinite(shuffled_energies)])

        # fit models
        dap_lognorm = lognorm.fit(dap_energies)
        shuf_lognorm = lognorm.fit(shuffled_energies)
    except Exception as e:
        print("Exception occurred during distribution fitting:", e)
        print("dap_energies:", dap_energies)
        print("shuffled_energies:", shuffled_energies)

        # Handle invalid values
        dap_energies = dap_energies[np.isfinite(dap_energies)]
        shuffled_energies = shuffled_energies[np.isfinite(shuffled_energies)]
        
        print("Filtered dap_energies:", dap_energies)
        print("Filtered shuffled_energies:", shuffled_energies)

        dap_lognorm = lognorm.fit(dap_energies)
        shuf_lognorm = lognorm.fit(shuffled_energies)

    return dap_lognorm, shuf_lognorm


def get_range(energies, shuffled_energies):
    return(min(energies.min(), shuffled_energies.min()), max(energies.max(), shuffled_energies.max()))

def log_prob_function(x, dap_lognorm, shuf_lognorm):
    true_pdf = lognorm.pdf(x, *dap_lognorm)
    bg_pdf = lognorm.pdf(x, *shuf_lognorm)
    # fix 0 values
    true_pdf[true_pdf == 0] = 1e-10
    bg_pdf[bg_pdf == 0] = 1e-10
    return np.log(true_pdf/bg_pdf), true_pdf, bg_pdf

def divide_range(start,end, divisions):
    res = []
    for i in range(divisions):
        div = start + i * (end-start) /divisions
        res.append(div)
    return res

def get_thresholds(x,y, num_divisions):
    y_greater_than_zero = y > -0.1
    
    false_idx = np.where(y_greater_than_zero == False)[0][0]  # Get the first occurrence of False
    if false_idx == 0: # If first occurence is 0
        true_idx = np.where(y_greater_than_zero == True)[0][0]
        y_greater_than_zero[:true_idx + 1] = True
        false_idx = np.where(y_greater_than_zero == False)[0][0]
    # Slice the array up to the first False
    y_first_zero = y[:false_idx]

    idx = (np.abs(y_first_zero - 0)).argmin(0)
    start = x[idx]
    end = min(x)
    return divide_range(start, end, num_divisions)

def bhattacharyya_coefficient(d1, d2, xmin, xmax):
    func = lambda x: np.sqrt(lognorm.pdf(x, *d1) * lognorm.pdf(x, *d2))
    return round(quad(func, xmin, xmax)[0],5)