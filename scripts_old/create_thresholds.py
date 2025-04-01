import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import os
from utils import *
from thresholds import *

random.seed(123)

TF= sys.argv[1]
DATA_FOLDER = '../data/' 
OUT_FILE = DATA_FOLDER + 'thresholds2.csv'
PLOT_FOLDER = DATA_FOLDER +'thresholds_plots/'
if not os.path.exists(PLOT_FOLDER): os.makedirs(PLOT_FOLDER)



# Imports
dap_seqs = import_seqs(f'{DATA_FOLDER}dap_fasta/{TF}.dap_peaks.fasta')
pwm = import_pwm(TF)
gc_dict = create_gc_dict()

# Calculate energies
dap_energies, shuffled_energies = get_dap_and_shuffled_energies(dap_seqs, pwm, gc_dict)

# get analysis stats
len_pwm = len(pwm['A'])
n_seqs = len(shuffled_energies)

# Create energy functions
dap_lognorm, shuf_lognorm = fit_distributions(dap_energies, shuffled_energies)

min_val, max_val = get_range(dap_energies, shuffled_energies)

x = np.linspace(min_val, max_val, 1000)
y, true_pdf, bg_pdf = log_prob_function(x, dap_lognorm, shuf_lognorm)
b_coeffcient = bhattacharyya_coefficient(dap_lognorm, shuf_lognorm, min(x), max(x))

thresholds = get_thresholds(x,y,3)

fig, ax1, ax2 = threshold_plot(TF,pwm,x,y,true_pdf,bg_pdf,dap_energies,shuffled_energies,thresholds)

fig.savefig(PLOT_FOLDER+f'{TF}.thresplot.png')

with open(OUT_FILE, 'a') as f:
    f.write(f'{TF},'
            f'{float(thresholds[2])},{float(thresholds[1])},{float(thresholds[0])},'
            f'{float(min_val)},{float(max_val)},'
            f'({float(dap_lognorm[0])}, {float(dap_lognorm[1])}, {float(dap_lognorm[2])}),'
            f'({float(shuf_lognorm[0])}, {float(shuf_lognorm[1])}, {float(shuf_lognorm[2])}),'
            f'{len_pwm},{n_seqs},{float(b_coeffcient)}\n')