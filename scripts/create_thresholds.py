import sys
import os
import argparse
from config import DATA_FOLDER, THRESHOLDS_FILE, PEAKS_SEQ_FOLDER
from thresholds import *

THRES_PLOT_FOLDER = os.path.join(DATA_FOLDER, 'thresholds_plots')

def process_tf(tf_name):
    """Processes a transcription factor to calculate and save thresholds.

    Args:
        tf_name (str): The name of the transcription factor.
    """
    if not os.path.exists(THRES_PLOT_FOLDER):
        os.makedirs(THRES_PLOT_FOLDER)

    dap_seqs = fasta_to_dict(os.path.join(PEAKS_SEQ_FOLDER, f'{tf_name}.dap_peaks.fasta'))
    pwm = pwm_from_tf_name(tf_name)
    gc_dict = create_gc_dict()

    dap_energies, shuffled_energies = get_dap_and_shuffled_energies(dap_seqs, pwm, gc_dict)

    len_pwm = len(pwm['A'])
    n_seqs = len(shuffled_energies)

    dap_lognorm, shuf_lognorm = fit_distributions(dap_energies, shuffled_energies)

    min_val, max_val = get_range(dap_energies, shuffled_energies)

    x = np.linspace(min_val, max_val, 1000)
    y, true_pdf, bg_pdf = log_prob_function(x, dap_lognorm, shuf_lognorm)
    b_coeffcient = bhattacharyya_coefficient(dap_lognorm, shuf_lognorm, min(x), max(x))

    thresholds = get_thresholds(x, y, 3)

    fig, ax1, ax2 = threshold_plot(tf_name, pwm, x, y, true_pdf, bg_pdf, dap_energies, shuffled_energies, thresholds)

    fig.savefig(os.path.join(THRES_PLOT_FOLDER, f'{tf_name}.thresplot.png'))

    with open(THRESHOLDS_FILE, 'a') as f:
        f.write(
            f'{tf_name},'
            f'{float(thresholds[2])},{float(thresholds[1])},{float(thresholds[0])},'
            f'{float(min_val)},{float(max_val)},'
            f'({float(dap_lognorm[0])}, {float(dap_lognorm[1])}, {float(dap_lognorm[2])}),'
            f'({float(shuf_lognorm[0])}, {float(shuf_lognorm[1])}, {float(shuf_lognorm[2])}),'
            f'{len_pwm},{n_seqs},{float(b_coeffcient)}\n'
        )

def main():
    """Main function to parse command-line arguments and process the transcription factor."""

    parser = argparse.ArgumentParser(description="Calculate and save thresholds for a given transcription factor.")
    parser.add_argument("-tf", "--tf_name", required=True, help="The name of the transcription factor.")

    args = parser.parse_args()

    process_tf(args.tf_name)

if __name__ == "__main__":
    main()