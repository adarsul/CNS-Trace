import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from scipy.integrate import quad
from reconstruct import fasta_to_dict
from model_utils import createPWMdb, create_gc_dict
from getBindingEnergy import getBindingEnergy
from config import PWM_FOLDER

random.seed(123)

def pwm_from_tf_name(TF, pwm_folder = PWM_FOLDER):
    """
    Retrieves a Position Weight Matrix (PWM) dictionary from a database using a transcription factor (TF) name.

    This function creates a PWM database from files within a specified folder and then retrieves
    the PWM dictionary associated with the given TF name.

    Args:
        TF (str): The name of the transcription factor for which to retrieve the PWM.
        pwm_folder (str, optional): The path to the folder containing PWM files.
                                   Defaults to the value of the global variable PWM_FOLDER.

    Returns:
        dict: The PWM dictionary corresponding to the given TF name.
    """
    pwm_db = createPWMdb(pwm_folder)
    return pwm_db[TF]


def shuffle(s:str):
    """
    Takes a string and returns a shuffled version of it

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

def threshold_plot(TF,pwm,x,y,true_pdf,bg_pdf,energies,shuffled_energies,thres_lines):
    """
    Generates a plot visualizing energy distributions and threshold determination for a given transcription factor (TF).

    This function creates a figure with two y-axes: one for displaying energy score distributions (histograms and fitted PDFs)
    and the other for showing the log probability ratio (S(E)) used for thresholding.

    Args:
        TF (str): The name of the transcription factor.
        pwm (dict): A dictionary representing the Position Weight Matrix (PWM) for the TF.
        x (array-like): An array of x-values (energy scores) for plotting fitted PDFs and S(E).
        y (array-like): An array of y-values representing the log probability ratio (S(E)).
        true_pdf (array-like): An array representing the probability density function (PDF) of energies from DAP-seq data.
        bg_pdf (array-like): An array representing the background PDF of energies from shuffled sequences.
        energies (array-like): An array of energy scores from DAP-seq data.
        shuffled_energies (array-like): An array of energy scores from shuffled sequences.
        thres_lines (list or array-like): A list of x-values representing threshold points.

    Returns:
        tuple: A tuple containing the figure (fig), the primary axes (ax1), and the secondary axes (ax2).
               (fig, ax1, ax2)

    Example:
        >>> fig, ax1, ax2 = threshold_plot("MyTF", pwm_data, x_values, y_values, true_pdf_data, bg_pdf_data, energies_data, shuffled_energies_data, threshold_points)
        >>> fig.show()

    Plots:
        - Histograms of energy scores from DAP-seq and shuffled sequences.
        - Fitted probability density functions (PDFs) for DAP-seq (P(E)) and shuffled (Q(E)) energies, with filled areas.
        - Log probability ratio (S(E)) plot, showing threshold points as vertical dashed lines.

    Visualizations:
        - The x-axis represents the energy score (E).
        - The primary y-axis (left) represents the density of energy scores.
        - The secondary y-axis (right) represents the log probability ratio (S(E)).
        - The plot includes a legend for all displayed data.
        - The plot title includes the TF name, the number of shuffled energies, and the k-mer length (from the PWM).
    """
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
    for x_val in thres_lines:
        plt.axvline(x_val, color= 'k', linestyle = 'dashed')

    ax1.set_xlabel('E (Energy Score)')
    ax1.set_ylabel('Density')
    ax2.set_ylabel('S(E)')

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


def fit_distributions(dap_energies, shuffled_energies):
    """
    Fits log-normal distributions to energy scores from DAP-seq and shuffled sequences.

    This function attempts to fit log-normal distributions to the provided energy score data.
    It includes error handling to manage potential issues with invalid (non-finite) values
    in the input arrays. If invalid values are detected, they are filtered out, and the fitting
    process is retried.

    Args:
        dap_energies (numpy.ndarray): An array of energy scores from DAP-seq data.
        shuffled_energies (numpy.ndarray): An array of energy scores from shuffled sequences.

    Returns:
        tuple: A tuple containing the fitted log-normal distribution parameters for DAP-seq
               and shuffled energies, respectively. Each parameter set is a tuple returned by
               `scipy.stats.lognorm.fit()`.

    Raises:
        Exception: If an unexpected error occurs during the distribution fitting process.
                   In this case, the function prints error information and attempts to recover
                   by filtering invalid values.

    Example:
        >>> dap_params, shuffled_params = fit_distributions(dap_energy_scores, shuffled_energy_scores)
        >>> if dap_params and shuffled_params:
        ...     print("Distributions fitted successfully.")
        ...     # Access fitted parameters: dap_params[0], dap_params[1], dap_params[2]
        ... else:
        ...     print("Failed to fit distributions.")

    Error Handling:
        - Checks for non-finite values (NaN, inf) in the input arrays.
        - Prints error messages if invalid values are found, indicating the problematic values.
        - Filters out non-finite values before attempting to fit the distributions again.
        - Prints the filtered arrays to aid in debugging.
    """

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
    """
    Determines the overall range of energy scores from two input arrays.

    This function calculates the minimum and maximum values across two arrays of energy scores,
    effectively finding the combined range of the input data.

    Args:
        energies (numpy.ndarray or array-like): An array of energy scores.
        shuffled_energies (numpy.ndarray or array-like): An array of shuffled energy scores.

    Returns:
        tuple: A tuple containing the minimum and maximum energy scores found in both input arrays.
               (min_energy, max_energy)

    Example:
        >>> dap_energies = np.array([-1.5, -0.8, 0.2, 1.0])
        >>> shuffled_energies = np.array([-2.0, -1.2, 0.5, 1.8])
        >>> min_val, max_val = get_range(dap_energies, shuffled_energies)
        >>> print(f"Minimum energy: {min_val}, Maximum energy: {max_val}")
        Minimum energy: -2.0, Maximum energy: 1.8
    """
    return(min(energies.min(), shuffled_energies.min()), max(energies.max(), shuffled_energies.max()))

def log_prob_function(x, dap_lognorm, shuf_lognorm):
    """
    Calculates the log-likelihood ratio for energy scores, indicating their likelihood of belonging
    to the DAP-seq (true) distribution versus the shuffled (background) distribution.

    This function computes the log probability ratio (log(P(E)/Q(E))), where P(E) is the probability
    density of an energy score 'E' in the DAP-seq distribution and Q(E) is the probability density
    in the shuffled distribution. A positive log-likelihood ratio suggests that the energy score is
    more likely to originate from the DAP-seq data, while a negative ratio suggests it's more likely
    from the shuffled data.

    Args:
        x (numpy.ndarray): An array of energy scores (x-values) for which to calculate the log probability ratio and PDFs.
        dap_lognorm (tuple): A tuple containing the parameters of the fitted log-normal distribution for DAP-seq energies.
                             (shape, location, scale).
        shuf_lognorm (tuple): A tuple containing the parameters of the fitted log-normal distribution for shuffled energies.
                              (shape, location, scale).

    Returns:
        tuple: A tuple containing:
            - log_prob (numpy.ndarray): An array of log probability ratios (log(P(E)/Q(E))).
                                        Values > 0 indicate a higher likelihood of belonging to DAP-seq,
                                        values < 0 indicate a higher likelihood of belonging to shuffled data.
            - true_pdf (numpy.ndarray): An array of probability density values for DAP-seq energies (P(E)).
            - bg_pdf (numpy.ndarray): An array of probability density values for shuffled energies (Q(E)).

    Context:
        This function is used to assess the likelihood that an observed energy score originates from
        a specific binding event (DAP-seq) versus random background noise (shuffled data).
        By comparing the probability densities of the two distributions, we can determine the relative
        likelihood of an energy score belonging to each.

    Example:
        >>> energy_scores = np.linspace(-3, 3, 100)
        >>> dap_params = (1.0, 0.0, 1.0)  # Example parameters
        >>> shuffled_params = (0.8, 0.0, 1.2) # Example parameters
        >>> log_ratio, dap_pdf, shuffled_pdf = log_prob_function(energy_scores, dap_params, shuffled_params)
        >>> # log_ratio > 0 indicates energies more likely from DAP-seq
        >>> # log_ratio < 0 indicates energies more likely from shuffled data
    """
    true_pdf = lognorm.pdf(x, *dap_lognorm)
    bg_pdf = lognorm.pdf(x, *shuf_lognorm)
    # fix 0 values
    true_pdf[true_pdf == 0] = 1e-10
    bg_pdf[bg_pdf == 0] = 1e-10
    return np.log(true_pdf/bg_pdf), true_pdf, bg_pdf

def divide_range(start,end, intervals):
    """
    Divides a numerical range into a specified number of equally spaced points.

    This function generates a list of numerical values that divide the range between 'start' and 'end'
    into equal intervals.

    Args:
        start (float or int): The starting value of the range.
        end (float or int): The ending value of the range.
        intervals (int): The number of intervals to create within the range.

    Returns:
        list: A list of numerical values representing the division points, including the 'start' value.

    Example:
        >>> divide_range(0, 10, 5)
        [0.0, 2.0, 4.0, 6.0, 8.0]
        >>> divide_range(1, 5, 3)
        [1.0, 2.3333333333333335, 3.6666666666666665]
    """
    
    res = []
    for i in range(intervals):
        div = start + i * (end-start) /intervals
        res.append(div)
    return res

def get_thresholds(x,y, num_divisions):
    """
    Determines threshold values based on the intersection of a function with a near-zero line.

    This function identifies a range of x-values where the corresponding y-values transition from positive to negative.
    It then finds the x-value where y is closest to zero within this range and uses this point as a starting point.
    Finally, it divides the range between this starting point and the minimum x-value into a specified number of divisions
    to generate threshold values.

    Args:
        x (numpy.ndarray): An array of x-values.
        y (numpy.ndarray): An array of y-values corresponding to the x-values.
        num_divisions (int): The number of threshold values to generate.

    Returns:
        list: A list of threshold values (x-values).

    Context:
        This function isused to identify energy thresholds based on the log probability ratio (S(E)).
        The goal is to find the x points where S(E) crosses or approaches zero, which indicates a transition
        from energies more likely to belong to the DAP-seq distribution to those more likely to belong to the
        shuffled distribution.

    Algorithm:
        1.  Identify the first index where y becomes less than -0.1.
        2.  Handle the edge case where the first y value is already below -0.1.
        3.  Slice the y-array up to the first negative value.
        4.  Find the index where y is closest to zero within the sliced array.
        5.  Use the corresponding x-value as the starting point.
        6.  Divide the range between the starting point and the minimum x-value into 'num_divisions' intervals.
        7.  Return the list of division points as threshold values.
    """
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
    """
    Calculates the Bhattacharyya coefficient between two log-normal distributions.

    The Bhattacharyya coefficient measures the overlap between two probability distributions.
    It provides a measure of similarity between the distributions, with higher values indicating
    greater overlap.

    Args:
        d1 (tuple): Parameters (shape, location, scale) of the first log-normal distribution.
        d2 (tuple): Parameters (shape, location, scale) of the second log-normal distribution.
        xmin (float): The lower bound of the integration range.
        xmax (float): The upper bound of the integration range.

    Returns:
        float: The calculated Bhattacharyya coefficient, rounded to 5 decimal places.

    Example:
        >>> dist1_params = (1.0, 0.0, 1.0)
        >>> dist2_params = (1.2, 0.2, 1.1)
        >>> coefficient = bhattacharyya_coefficient(dist1_params, dist2_params, -3, 3)
        >>> print(coefficient)

    Calculation:
        The Bhattacharyya coefficient is calculated by integrating the square root of the product
        of the probability density functions (PDFs) of the two distributions over the specified range.
        The `scipy.integrate.quad` function is used for numerical integration.
    """
    func = lambda x: np.sqrt(lognorm.pdf(x, *d1) * lognorm.pdf(x, *d2))
    return round(quad(func, xmin, xmax)[0],5)