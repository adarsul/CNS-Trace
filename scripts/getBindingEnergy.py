from model_utils import *
import argparse
import math
import numpy as np
import os
from pathlib import Path

AMBIGUOUS_NUCLEOTIDES = {
    "R": ["A", "G"],  # Purine
    "Y": ["C", "T"],  # Pyrimidine
    "S": ["G", "C"],  # Strong interaction
    "W": ["A", "T"],  # Weak interaction
    "K": ["G", "T"],  # Keto
    "M": ["A", "C"],  # Amino
    "B": ["C", "G", "T"],  # Not A
    "D": ["A", "G", "T"],  # Not C
    "H": ["A", "C", "T"],  # Not G
    "V": ["A", "C", "G"],  # Not T
    "N": ["A", "C", "G", "T"],
    "X": ["A", "C", "G", "T"]# Any base
}


def initialize_output_file(cns_id, output_folder, output_file_suffix):
    """Creates or clears an output file and returns its path.

    This function ensures that the specified output directory exists,
    constructs the full output file path, and either creates a new, empty
    file or truncates an existing file to zero length.

    Args:
        cns_id (str): The CNS ID. This is used to construct the filename.
        output_folder (str): The path to the directory where the output file
            should be created.
        output_file_suffix (str): The suffix (including the extension, e.g., ".txt")
            to be added to the CNS ID to form the filename.

    Returns:
        str: The full path to the created or cleared output file.

    Example:
        >>> output_file = initialize_output_file("CNS42", "/path/to/results", "_results.txt")
        >>> print(output_file)
        /path/to/results/CNS42_results.txt

    Notes:
        - If the `output_folder` does not exist, it will be created.
        - If a file with the constructed filename already exists, it will be
          truncated (emptied).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    filename = os.path.join(output_folder, cns_id + output_file_suffix)
    open(filename, 'w').close()  # Create or truncate the file
    return filename



def f(pwm,b,i):
    """
    Retrieves the weight of a specific nucleotide at a given position within a Position Weight Matrix (PWM).

    Args:
        pwm (dict): A dictionary representing the PWM. The keys are nucleotides ('A', 'C', 'G', 'T'),
                    and the values are lists of weights corresponding to each position.
        b (str): The nucleotide ('A', 'C', 'G', or 'T') for which to retrieve the weight.
        i (int): The position (0-indexed) within the PWM for which to retrieve the weight.

    Returns:
        float: The weight of the specified nucleotide at the given position in the PWM.

    Raises:
        KeyError: If the nucleotide 'b' is not present in the pwm.
        IndexError: If the position 'i' is out of range for the nucleotide's weight list.

    Example:
        pwm = {'A': [0.2, 0.1], 'C': [0.3, 0.4], 'G': [0.1, 0.2], 'T': [0.4, 0.3]}
        weight = f(pwm, 'C', 1)  # Returns 0.4
    """
    return pwm[b][i]

def p(b, species, gc_dict):
    """
    Retrieves the background frequency of a nucleotide based on the specified species or the mean background frequency.

    Args:
        b (str): The nucleotide ('A', 'C', 'G', or 'T') for which to retrieve the background frequency.
        species (str or None): The species name. If None or an empty string, the mean background frequency is used.
                               If a species name is provided, the species-specific background frequency is used.
        gc_dict (dict): A dictionary containing background frequencies. It should have a 'Mean' key with the mean
                        frequencies and optional species-specific keys, each holding a dictionary of nucleotide frequencies.

    Returns:
        float: The background frequency of the specified nucleotide.

    Raises:
        KeyError: If the nucleotide 'b' is not found in the 'Mean' or the specified species' dictionary.

    Example:
        gc_dict = {
            'Plant_A': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
            'Plant_B': {'A': 0.28, 'C': 0.22, 'G': 0.22, 'T': 0.28},
            'Plant_C': {'A': 0.30, 'C': 0.20, 'G': 0.20, 'T': 0.30}
        }
        freq_mean = p('A', None, gc_dict)  # Returns 0.25
        freq_plant_a = p('C', 'Plant_A', gc_dict)  # Returns 0.22
        freq_default = p('G', 'InvalidSpecies', gc_dict) #Returns 0.25
    """
    if not species:
        return(gc_dict['Mean'][b])
    if species:
        try:
            bg_freq = gc_dict[species]
            return bg_freq[b]
        except:
            return(gc_dict['Mean'][b])

def g(pwm, b, i, species, gc_dict):
    """
    Calculates an energy score for a single position within a DNA sequence, based on the likelihood ratio
    of a base appearing in a Transcription Factor Binding Site (TFBS) motif versus its background frequency in the genome.

    Args:
        pwm (dict): A dictionary representing the Position Weight Matrix (PWM) of the TFBS motif.
                    Keys are nucleotides ('A', 'C', 'G', 'T'), and values are lists of weights for each position.
        b (str): The nucleotide ('A', 'C', 'G', 'T') at the given position.
        i (int): The position (0-indexed) within the sequence and PWM.
        species (str or None): The species name for background frequency calculation. If None, the mean background
                               frequency is used.
        gc_dict (dict): A dictionary containing background frequencies. It should have a 'Mean' key with mean
                        frequencies and optional species-specific keys, each holding a dictionary of nucleotide frequencies.

    Returns:
        float: The energy score for the given nucleotide at the specified position. A higher score indicates a stronger
               preference for that nucleotide at that position in the TFBS motif compared to the genomic background.

    Raises:
        KeyError: If the nucleotide 'b' is not found in the PWM or gc_dict.
        IndexError: If the position 'i' is out of range for the PWM.
        ZeroDivisionError: If the background frequency p(b, species, gc_dict) is zero.

    Example:
        pwm = {'A': [0.2, 0.1], 'C': [0.3, 0.4], 'G': [0.1, 0.2], 'T': [0.4, 0.3]}
        gc_dict = {
            'Mean': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
            'Athaliana': {'A': 0.28, 'C': 0.22, 'G': 0.22, 'T': 0.28}
        }
        energy_score = g(pwm, 'C', 1, 'Athaliana', gc_dict)
    """
    return -math.log2(f(pwm,b,i) / p(b, species, gc_dict))

def get_pos_energy(pwm, b, i, species, gc_dict):
    """
    Calculates the energy score for a single position in a DNA sequence, handling ambiguous nucleotides.

    If the nucleotide at the given position is ambiguous (e.g., 'R' for A or G), the function calculates the
    energy score for each possible nucleotide and returns the sum of these scores. This accounts for uncertainty
    in the nucleotide identity.

    Args:
        pwm (dict): A dictionary representing the Position Weight Matrix (PWM) of the TFBS motif.
                    Keys are nucleotides ('A', 'C', 'G', 'T'), and values are lists of weights for each position.
        b (str): The nucleotide at the given position. It can be a standard nucleotide ('A', 'C', 'G', 'T') or
                 an ambiguous nucleotide (e.g., 'R', 'Y', 'M', 'K', 'S', 'W', 'H', 'B', 'V', 'D', 'N').
        i (int): The position (0-indexed) within the sequence and PWM.
        species (str or None): The species name for background frequency calculation. If None, the mean background
                               frequency is used.
        gc_dict (dict): A dictionary containing background frequencies. It should have a 'Mean' key with mean
                        frequencies and optional species-specific keys, each holding a dictionary of nucleotide frequencies.

    Returns:
        float: The energy score for the given position. If the nucleotide is ambiguous, it returns the sum of
               energy scores for the possible nucleotides. Otherwise, it returns the energy score for the
               specified nucleotide.

    Raises:
        KeyError: If a nucleotide (or part of an ambiguous nucleotide) is not found in the PWM or gc_dict.
        IndexError: If the position 'i' is out of range for the PWM.
        ZeroDivisionError: If a background frequency is zero.

    Example:
        AMBIGUOUS_NUCLEOTIDES = {
            'R': ['A', 'G'],
            'Y': ['C', 'T'],
            # ... other ambiguous nucleotide definitions ...
        }
        pwm = {'A': [0.2, 0.1], 'C': [0.3, 0.4], 'G': [0.1, 0.2], 'T': [0.4, 0.3]}
        gc_dict = {
            'Mean': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
            'Human': {'A': 0.28, 'C': 0.22, 'G': 0.22, 'T': 0.28}
        }
        energy_score_A = get_pos_energy(pwm, 'A', 0, 'Athaliana', gc_dict)
        energy_score_R = get_pos_energy(pwm, 'R', 1, 'Athaliana', gc_dict) #will sum g('A') and g('G') at position 1.
    """
    if b in AMBIGUOUS_NUCLEOTIDES.keys():
        return sum([g(pwm, nt, i, species,gc_dict) for nt in AMBIGUOUS_NUCLEOTIDES[b]])
    else:
        return g(pwm, b, i, species, gc_dict)

def get_energy(seq, pwm, species, gc_dict):
    """
    Calculates the total binding energy of a DNA sequence based on a Position Weight Matrix (PWM).

    This function iterates through each position in the input sequence, calculates the energy score
    for that position using the provided PWM and background frequencies, and returns the sum of all
    position-wise energy scores.

    Args:
        seq (str): The DNA sequence for which to calculate the binding energy.
        pwm (dict): A dictionary representing the PWM of the TFBS motif.
                    Keys are nucleotides ('A', 'C', 'G', 'T') or ambiguous nucleotides, and values are
                    lists of weights for each position.
        species (str or None): The species name for background frequency calculation. If None, the mean background
                               frequency is used.
        gc_dict (dict): A dictionary containing background frequencies. It should have a 'Mean' key with mean
                        frequencies and optional species-specific keys, each holding a dictionary of nucleotide frequencies.

    Returns:
        float: The total binding energy of the input sequence.

    Raises:
        KeyError: If a nucleotide in the sequence or PWM is not found in the gc_dict.
        IndexError: If the length of the sequence does not match the PWM's length.
        ZeroDivisionError: If a background frequency is zero.

    Example:
        pwm = {'A': [0.2, 0.1], 'C': [0.3, 0.4], 'G': [0.1, 0.2], 'T': [0.4, 0.3]}
        gc_dict = {
            'Mean': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
            'Athaliana': {'A': 0.28, 'C': 0.22, 'G': 0.22, 'T': 0.28}
        }
        seq = "ACGT"
        energy = get_energy(seq, pwm, "Athaliana", gc_dict)
    """
    
    vec = np.zeros(len(seq))
    for i in range(len(seq)):
        base = seq[i]
        vec[i] = (get_pos_energy(pwm,base,i, species, gc_dict))
    return vec.sum()

def compliment(N):
    """
    Returns the complementary nucleotide for a given nucleotide.

    This function takes a single nucleotide character as input and returns its
    complementary nucleotide. It handles standard DNA nucleotides (A, T, G, C)
    and also 'X' or 'N' for indefinite nucleotides, returning them unchanged.

    Args:
        N (str): The nucleotide character (A, T, G, C, X, or N). Case-insensitive.

    Returns:
        str: The complementary nucleotide.

    Raises:
        ValueError: If the input nucleotide is not one of 'A', 'T', 'G', 'C', 'X', or 'N'.

    Example:
        compliment('a')  # Returns 'T'
        compliment('G')  # Returns 'C'
        compliment('x') #Returns 'X'
    """
    N = N.upper()
    if N == 'A':
        return 'T'
    elif N == 'T':
        return 'A'
    elif N == 'G':
        return 'C'
    elif N == 'C':
        return 'G'
    elif N == 'X' or N == 'N':
        return N

    
def rc_pwm(pwm):
    """
    Calculates the reverse complement of a Position Weight Matrix (PWM).

    This function takes a PWM dictionary as input and returns a new PWM dictionary
    representing the reverse complement of the input. The keys (nucleotides) are
    complemented, and the values (lists of weights) are reversed.

    Args:
        pwm (dict): A dictionary representing the PWM. Keys are nucleotides ('A', 'C', 'G', 'T'),
                    and values are lists of weights corresponding to each position.

    Returns:
        dict: A dictionary representing the reverse complement of the input PWM.

    Example:
        pwm = {'A': [0.1, 0.2], 'C': [0.3, 0.4], 'G': [0.5, 0.6], 'T': [0.7, 0.8]}
        rc_pwm(pwm)  # Returns {'T': [0.8, 0.7], 'G': [0.6, 0.5], 'C': [0.4, 0.3], 'A': [0.2, 0.1]}
    """
    return {compliment(x): list(reversed(y)) for x,y in pwm.items()}

def add_N_padding (seq, amount):
    """
    Adds 'N' padding to both ends of a DNA sequence.

    This function takes a DNA sequence and an integer amount as input. It returns a new
    sequence with 'N' characters added to both the beginning and the end of the
    original sequence. The number of 'N' characters added to each end is determined
    by the 'amount' parameter.

    Args:
        seq (str): The DNA sequence to which 'N' padding will be added.
        amount (int): The number of 'N' characters to add to each end of the sequence.

    Returns:
        str: The DNA sequence with 'N' padding added to both ends.

    Example:
        add_N_padding("ATGC", 3)  # Returns "NNNATGCNNN"
        add_N_padding("CG", 0)    # Returns "CG"
    """
    return amount * 'N' + seq + amount * 'N'

def short_seq_fix(seq, len_pwm):
    """
    Pads a short DNA sequence with 'N' characters to ensure it's long enough for PWM analysis.

    This function takes a DNA sequence and the length of a Position Weight Matrix (PWM)
    as input. It calculates the amount of 'N' padding needed to be added to both ends
    of the sequence to ensure that the sequence is at least as long as the PWM. 

    Args:
        seq (str): The DNA sequence to be padded.
        len_pwm (int): The length of the PWM.

    Returns:
        str: The padded DNA sequence.

    Example:
        short_seq_fix("ATGC", 6)  # Returns "NNNATGCNNN"
        short_seq_fix("CG", 3)    # Returns "NCGN"
    """
    amount = math.ceil(len_pwm / 2)
    return add_N_padding(seq, amount)

def part_seqs(seq, n):
    """
    Generates a list of overlapping substrings of a specified length from a given sequence.

    This function takes a string sequence and an integer length 'n' as input. It returns a list
    containing all overlapping substrings of length 'n' extracted from the input sequence.
    All substrings are converted to uppercase.

    Args:
        seq (str): The input sequence from which substrings will be generated.
        n (int): The desired length of each substring.

    Returns:
        list: A list of overlapping substrings of length 'n', all in uppercase.

    Example:
        part_seqs("atgc", 2)  # Returns ["AT", "TG", "GC"]
        part_seqs("abcdef", 3) # Returns ["ABC", "BCD", "CDE", "DEF"]
        part_seqs("A", 2) # returns []
    """
    i = 0
    partial_seqs = []
    while i + n <= len(seq):
        partial_seqs.append((seq[i:i+n].upper()))
        i += 1
    return partial_seqs

def valid_seq(seq):
    """
    Checks if a DNA sequence contains only valid nucleotide characters.

    This function iterates through the characters of a given DNA sequence and checks if each
    character is a valid nucleotide. Valid nucleotides are defined as standard DNA bases
    ('A', 'C', 'G', 'T') or ambiguous nucleotides (keys of the `AMBIGUOUS_NUCLEOTIDES` dictionary).

    Args:
        seq (str): The DNA sequence to be validated.

    Returns:
        bool: True if the sequence contains only valid nucleotides, False otherwise.

    Example:
        AMBIGUOUS_NUCLEOTIDES = {'R': ['A', 'G'], 'Y': ['C', 'T']} # Example ambiguous nucleotides
        valid_seq("ACGT")    # Returns True
        valid_seq("ACGTN")   # Assuming 'N' is not in AMBIGUOUS_NUCLEOTIDES, returns False
        valid_seq("ACGR")    # Returns True if 'R' is in AMBIGUOUS_NUCLEOTIDES
        valid_seq("ACGU")    # Returns False
    """
    for char in seq:
        if char in ['A','C','G','T'] + list(AMBIGUOUS_NUCLEOTIDES.keys()):
            continue
        else:
            return False
    return True


def get_energy_and_direction(seq, pwm, r_pwm, species, gc_dict):
    """
    Applies an energy function to a sequence and its reverse complement, and returns the lower energy value and its corresponding direction.

    This function calculates the binding energy of a given sequence with a Position Weight Matrix (PWM) and its reverse complement (r_pwm).
    It then determines which direction (forward or reverse) yields the lower binding energy and returns both the energy value and the direction.

    Args:
        seq (str): The DNA sequence to which the energy function is applied.
        pwm (dict): The Position Weight Matrix (PWM) for the forward direction.
        r_pwm (dict): The Position Weight Matrix (PWM) for the reverse complement direction.
        species (str): The species of the sequence, used for GC content adjustment.
        gc_dict (dict): A dictionary containing GC content information for the species.

    Returns:
        tuple: A tuple containing the lower binding energy (float) and the corresponding direction ('forward' or 'reverse').
    """
    forward = get_energy(seq, pwm, species, gc_dict)  
    reverse = get_energy(seq, r_pwm, species, gc_dict) 
    if forward <= reverse:
        return forward, 'forward'
    else:
        return reverse, 'reverse'

def calculate_bind_pos(i, extension, direction, len_pwm):
    """
    Calculates the binding position (0-based index) within a sequence, considering sequence extension and PWM length.

    This function calculates the starting position of a binding site within a sequence, taking into account any sequence extension (due to flanking site search),
    the direction of the binding (forward or reverse), and the length of the Position Weight Matrix (PWM).

    Args:
        i (int): The initial position index of the binding site.
        extension (int): The length of the sequence extension.
        direction (str): The direction of the binding ('forward' or 'reverse').
        len_pwm (int): The length of the Position Weight Matrix (PWM).

    Returns:
        int: The calculated 0-based index of the binding position.
    """
    if direction == 'reverse':
        return int(i + extension / 2 - len_pwm - 1)
    else:
        return int(i - extension / 2)

def calc_extension(seq):
    """
    Calculates the negative extension value based on the number of 'N' characters at the beginning of a sequence.

    This function determines the negative extension value by counting the number of 'N' characters at the beginning of a sequence.
    This is used to adjust binding positions when sequences have been extended with 'N' characters for flanking site analysis.

    Args:
        seq (str): The DNA sequence.

    Returns:
        int: The negative extension value (0 if no 'N' characters are found).
    """
    count = 0
    for i in seq:
        if i == 'N':
            count += 1
        else:
            return 0 - count

def get_bind_pos(seq, i, extension, direction, len_pwm):
    """
    Gets the index position of a binding sequence within a sequence, handling negative positions due to extensions.

    This function calculates the binding position using `calculate_bind_pos` and adjusts it if the calculated position is negative,
    indicating that the binding site is within the extended portion of the sequence.

    Args:
        seq (str): The DNA sequence.
        i (int): The initial position index of the binding site.
        extension (int): The length of the sequence extension.
        direction (str): The direction of the binding ('forward' or 'reverse').
        len_pwm (int): The length of the Position Weight Matrix (PWM).

    Returns:
        int: The adjusted 0-based index of the binding position.
    """
    pos = calculate_bind_pos(i, extension, direction, len_pwm)
    if pos <= 0:
        pos = calc_extension(seq)
    return pos

def get_binding_category(energy, threshold):
    """
    Determines the binding category based on the energy and threshold.
    """
    if energy > threshold['weak_thres']:
        return 'None'
    elif energy > threshold['medium_thres']:
        return 'Weak'
    elif energy > threshold['strong_thres']:
        return 'Medium'
    else:
        return 'Strong'

def get_distance_from_threshold(energy, threshold, category):
    """
    Calculates the distance of the energy from the respective threshold.
    """
    if category == 'None':
        return energy - threshold['weak_thres']
    elif category == 'Weak':
        return energy - threshold['medium_thres']
    elif category == 'Medium':
        return energy - threshold['strong_thres']
    else:  # category == 'Strong'
        return energy - threshold['strong_thres']

def classify_binding_energy(energy, tf, thres_db, calc_dist_to_thres = False):
    """
    Classifies the binding energy into a category and calculates the distance from the threshold.
    """
    tf_thres = thres_db[tf]
    category = get_binding_category(energy, tf_thres)
    
    if calc_dist_to_thres:
        distance = get_distance_from_threshold(energy, tf_thres, category)
        return category, distance
    
    return category


def getBindingEnergy(seq: str, pwm: dict, gc_dict: dict, species=None, energy_only=False) -> tuple[float, int, str] or float or None:
    """
    Calculates the minimum binding energy of a sequence to a Position Weight Matrix (PWM) and identifies the corresponding binding site.

    This function processes a DNA sequence to find the subsequence with the lowest binding energy relative to a given PWM.
    It handles sequence length adjustments, calculates energies for both forward and reverse complement directions,
    and returns the minimum energy, its position within the original sequence, and the subsequence itself.

    Args:
        seq (str): The DNA sequence to analyze. Any '-' characters are removed.
        pwm (dict): The Position Weight Matrix (PWM) as a dictionary, where keys are nucleotides ('A', 'C', 'G', 'T')
                    and values are lists representing the positional weights.
        gc_dict (dict): A dictionary containing GC content information for the specified species, used for energy calculations.
        species (str, optional): The species of the sequence, used for GC content adjustment in energy calculations. Defaults to None.
        energy_only (bool, optional): If True, returns only the minimum binding energy (float). Defaults to False.

    Returns:
        tuple[float, int, str] or float or None:
            - If `energy_only` is False: A tuple containing the minimum binding energy (float),
              the 0-based index of the binding site's start position in the original sequence (int),
              and the subsequence with the minimum energy (str).
            - If `energy_only` is True: The minimum binding energy (float).
            - None: If no valid binding site is found (e.g., all subsequences are invalid).

    Note:
        - The function removes '-' characters from the input sequence.
        - If the input sequence is shorter than the PWM length, it is extended with 'N' characters.
        - The function utilizes `rc_pwm`, `short_seq_fix`, `part_seqs`, `valid_seq`, `get_energy_and_direction`, and `get_bind_pos` functions,
          which are assumed to be defined elsewhere.
        - A minimum energy of 9999 is used as an initial placeholder and indicates no valid binding site if returned.

    Example:
        >>> pwm = {'A': [0.25, 0.5], 'C': [0.25, 0.25], 'G': [0.25, 0.25], 'T': [0.25, 0.0]}
        >>> gc_dict = {'species1': 0.5}
        >>> seq = "ACGTACGT"
        >>> energy, pos, subseq = getBindingEnergy(seq, pwm, gc_dict, species='species1')
        >>> print(f"Energy: {energy}, Position: {pos}, Subsequence: {subseq}")
        Energy: -0.5, Position: 0, Subsequence: AC
    """
    seq = seq.replace('-', '')
    min_energy = 9999
    min_pos = None
    min_seq = ''
    # initialize reverse compliment PWM
    r_pwm = rc_pwm(pwm)  # Assuming rc_pwm is defined elsewhere

    # If sequence is too short for PWM, extend accordingly
    len_pwm = len(pwm['A'])
    old_seq_len = len(seq)
    seq = short_seq_fix(seq, len_pwm)  # Assuming short_seq_fix is defined elsewhere
    extension = len(seq) - old_seq_len

    # for every partial seq in length of PWM
    for i, p in enumerate(part_seqs(seq, len_pwm)):  # Assuming part_seqs is defined elsewhere
        if valid_seq(p):  # check sequence validtiy # Assuming valid_seq is defined elsewhere
            # apply PWM and reverse comp PWM
            current_energy, direction = get_energy_and_direction(p, pwm, r_pwm, species, gc_dict)
        else:
            raise ValueError(f"Invalid subsequence found at position {i}: {p}")  # Raise error if invalid


        if current_energy < min_energy:
            min_energy = current_energy
            min_pos = get_bind_pos(p, i, extension, direction, len_pwm) # Assuming get_bind_pos is defined elsewhere
            min_seq = p

    # return the lowest energy
    if min_energy == 9999:
        return None

    if energy_only:
        return min_energy

    return min_energy, min_pos, min_seq

def main(input_file, output_file, cns_id_col, species_col, seq_col, print_no_bind=True, tf_names=None):
    """
    Processes binding results from an input CSV file and writes the results to an output CSV file.

    This function reads sequences from the input CSV, calculates binding energies, and categorizes
    binding events based on predefined PWM and threshold databases. It then writes the results,
    including CNS ID, species, transcription factor, binding energy, position, binding sequence,
    binding category, distance to threshold, and any other columns from the input, to the output CSV.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to the output CSV file.
        cns_id_col (int): Index of the CNS ID column in the input CSV (0-based).
        species_col (int): Index of the species column in the input CSV (0-based).
        seq_col (int): Index of the sequence column in the input CSV (0-based).
        print_no_bind (bool, optional): If True, includes rows with no binding ('None' category) in the output.
                                         If False, only includes rows with positive binding results. Defaults to True.
        tf_names (list, optional): A list of transcription factor names to use. 
                                     If None, all TFs in the database are used.
    """
    output_file_dir = Path(input_file).parent

    def binding_results(seq, tf, species, pwm_db, thres_db, gc_dict):
        """Calculates binding results for a given sequence and transcription factor."""
        energy, pos, bind_seq = getBindingEnergy(seq, pwm_db[tf], gc_dict, species=species)
        binding_category, dist_to_thres = classify_binding_energy(energy, tf, thres_db, True)
        return energy, pos, bind_seq, binding_category, dist_to_thres

    def write_faulty_row(row, path):
        """Writes a faulty row to a CSV file."""
        faulty_rows_file = os.path.join(path, 'faulty_rows.csv')
        with open(faulty_rows_file, 'a') as f_file:
            writer = csv.writer(f_file)
            writer.writerow(row)

    # Create databases
    pwm_db = createPWMdb()
    thres_db = importThresDB()
    gc_dict = create_gc_dict()

    if tf_names:
        pwm_db = filter_pwm_db(pwm_db, tf_names)

    tfs = list(pwm_db.keys())

    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        with open(output_file, 'w') as res_file:
            writer = csv.writer(res_file)
            for row in reader:
                try:
                    # Extract CNS ID, species, and sequence using the specified indices
                    cns_id = row[cns_id_col]
                    species = row[species_col]
                    seq = row[seq_col]
                    
                    # Get all other columns except CNS ID, species, and sequence
                    other_columns = [val for idx, val in enumerate(row) if idx not in {cns_id_col, species_col, seq_col}]

                    for tf in tfs:
                        energy, pos, bind_seq, binding_category, dist_to_thres = binding_results(seq, tf, species, pwm_db, thres_db, gc_dict)
                        if binding_category != 'None' or print_no_bind:
                            # Append additional columns to the result
                            res_row = [cns_id, species, tf, round(energy, 5), pos, bind_seq, binding_category, round(dist_to_thres, 5)] + other_columns
                            writer.writerow(res_row)
                        else:
                            continue
                except Exception as e:
                    write_faulty_row(row, output_file_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process binding results from input CSV file. Creates a file with TF binding energy model results.\n" +
                                                 "Result columns: ID, Species, Transcription Factor, Energy Score, Binding Position (in sequence), Binding Sequence, Binding Classification, Distance from threshold + any other columns in input.")
    parser.add_argument('-i', '--input', required=True, help="Path to the input CSV file.")
    parser.add_argument('-o', '--output', required=True, help="Path to the output CSV file.")
    parser.add_argument('--cns_id_col', type=int, default=0, help="Index of the CNS ID column in the input CSV.\nDefault = 1st")
    parser.add_argument('--species_col', type=int, default=1, help="Index of the species column in the input CSV.\nDefault = 2nd")
    parser.add_argument('--seq_col', type=int, default=6, help="Index of the sequence column in the input CSV.\nDefault = 7th")
    parser.add_argument('--positive_only', action='store_true', help="If given, only print rows with positive binding results.")
    parser.add_argument("-t", "--tfs", nargs='+', help="List of transcription factor names (space-separated) to use. If not provided, all TFs are used.")

    args = parser.parse_args()

    try:
        main(input_file=args.input,
             output_file=args.output,
             cns_id_col=args.cns_id_col,
             species_col=args.species_col,
             seq_col=args.seq_col,
             print_no_bind=not args.positive_only,
             tf_names=args.tfs)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)