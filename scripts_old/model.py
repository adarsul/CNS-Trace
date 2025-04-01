from utils import *
import math
import numpy as np
import csv
import argparse
import os

DATA_FOLDER = '../data/'
OUTPUT_FOLDER = '../output/'
ambiguous_nucleotides = {
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


def initialize_output_file(cns_id, output_folder, suffix):
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    filename = output_folder + cns_id + suffix
    open(filename, 'w').close() if os.path.exists(filename) else None
    return filename


def species_bg_freq(gc):
    at = 1 - gc
    return {'A': at/2, 'C': gc/2, 'G': gc/2, 'T': at/2}

def create_gc_dict(csv_file = 'gc_content.csv', data_folder = DATA_FOLDER):
    result_dict = {}
    with open(data_folder + csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if len(row) >= 2:  # Ensure at least two columns in the row
                gc = float(row[1])
                result_dict[row[0]] = species_bg_freq(gc)
    return result_dict

def createPWMdb(prefix = 'meme', data_folder = DATA_FOLDER ):
    """ 
    Creates a dictionary containing all PWMs and their names in a dictionary from working directory.
    
    Args:
        prefix (str): The prefix of the files containing PWM data
    
    Returns:
        pwm_db (dict): A dictionary where each key is the name of a TF and the value is a PWM in a dictionary
    """
    dir = data_folder + 'dap_pwms/'
    # initialize empty dictionary
    pwm_db = {} 
    # get list of meme files
    if dir:
        pwm_filenames = find_pwm_files(prefix, dir)
    else:    
        pwm_filenames = find_pwm_files(prefix)

    # iterate through meme files 
    for pwm_file in pwm_filenames: 
        # turn meme file into pwm dictionary
        pwm = parse_meme(dir + pwm_file)         
        # modify pwm file name into TF name
        pwm_name = pwm_name_from_file(pwm_file)
        # add element to dictionary dict['TF'] = pwm
        pwm_db[pwm_name] = pwm 
    
    return pwm_db

def importThresDB(thresholds_file = 'thresholds.csv', data_folder = DATA_FOLDER):
    thres_db = {}
    with open(data_folder + thresholds_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            thres_db[row[0]] = {'strong_thres' : float(row[1]), 'medium_thres': float(row[2]), 'weak_thres': float(row[3])}
    return thres_db

def f(pwm,b,i):
    return pwm[b][i]

def p(b, species, gc_dict):
    if not species:
        return(gc_dict['Mean'][b])
    if species:
        try:
            bg_freq = gc_dict[species]
            return bg_freq[b]
        except:
            return(gc_dict['Mean'][b])

def g(pwm, b, i, species, gc_dict):
    return -math.log2(f(pwm,b,i) / p(b, species, gc_dict))

def get_pos_energy(pwm, b, i, species, gc_dict):
    if b in ambiguous_nucleotides.keys():
        return sum([g(pwm, nt, i, species,gc_dict) for nt in ambiguous_nucleotides[b]])
    else:
        return g(pwm, b, i, species, gc_dict)

def get_energy(seq, pwm, species, gc_dict):
    """
    Takes a sequence, position weight-matrix (dictionary) and length of pwm sequences.
    returns the sum of binding energy of the whole sequence.
    """
    vec = np.zeros(len(seq))
    for i in range(len(seq)):
        base = seq[i]
        vec[i] = (get_pos_energy(pwm,base,i, species, gc_dict))
    return vec.sum()

def compliment(N):
    """
    Takes a single nucleotide and returns the complimentary nucleotide
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
    """Takes a PWM and returns it's reverse complimentary matrix"""
    return {compliment(x): list(reversed(y)) for x,y in pwm.items()}

def extend_sequence_Ns(seq, amount):
    return amount * 'N' + seq + amount * 'N'

def short_seq_fix(seq, len_pwm):
#    amount = math.ceil((len_pwm - len(seq))/2)
    amount = math.ceil(len_pwm / 2)
    return extend_sequence_Ns(seq, amount)

def part_seqs(seq, n):
    """
    Takes a string and returns a list of substrings of length n
    """
    i = 0
    partial_seqs = []
    while i + n <= len(seq):
        partial_seqs.append((seq[i:i+n].upper()))
        i += 1
    return partial_seqs

def valid_seq(seq):
    for char in seq:
        if char in ['A','C','G','T'] + list(ambiguous_nucleotides.keys()):
            continue
        else:
            return False
    return True

def unload_node(node, sequences):
    return node.name, sequences[node.name]

def get_energy_and_direction(seq, pwm, r_pwm, species, gc_dict):
    forward = get_energy(seq, pwm, species, gc_dict)
    reverse = get_energy(seq, r_pwm, species, gc_dict)
    if forward <= reverse:
        return forward, 'forward'
    else:
        return reverse, 'reverse'
    
def calculate_bind_pos(i, extension, direction, len_pwm):
    if direction == 'reverse':
        return int(i + extension/2 - len_pwm - 1)
    else:
        return int(i - extension/2)        

def calc_extension(seq):
    count = 0
    for i in seq:
        if i == 'N':
            count += 1
        else:
            return 0 - count

def get_bind_pos(seq, i, extension, direction, len_pwm):
    pos = calculate_bind_pos(i, extension, direction, len_pwm)
    if pos <= 0:
        pos = calc_extension(seq)
    return pos

def getBindingEnergy (seq: str, pwm: dict, gc_dict : dict, species = None, energy_only = False) -> float:
    """
    This function takes a sequence, splits it to sub-sequences of the same length as the PWM and calculates the energies of each sub sequence.

    Returns the energy of the sub-sequence with the lowest energy, the position of the subsequence (first nt is 0) and the subsequence.
    """
    seq = seq.replace('-','')
    min_energy = 9999
    min_pos = None
    min_seq = ''
    # initialize reverse compliment PWM
    r_pwm  = rc_pwm(pwm) 

    # If sequence is too short for PWM, extend accordingly
    len_pwm = len(pwm['A'])
    old_seq_len = len(seq)
    seq = short_seq_fix(seq, len_pwm)
    extension = len(seq) - old_seq_len
        
    # for every partial seq in length of PWM
    for i, p in enumerate(part_seqs(seq, len_pwm)): 
        if valid_seq(p):  # check sequence validtiy
            # apply PWM and reverse comp PWM 
            current_energy, direction = get_energy_and_direction(p, pwm, r_pwm, species, gc_dict)
        else:
            continue

        if current_energy < min_energy:
            min_energy = current_energy
            min_pos = get_bind_pos(p, i, extension, direction, len_pwm)
            min_seq = p
    
    #return the lowest energy
    if min_energy == 9999:
        return None
    
    if energy_only:
        return min_energy
    
    return min_energy, min_pos, min_seq

def is_node_name(string):
    if string.startswith('N') and string[1].isdigit():
        return True

def getCNSbindProfile(seq: str, pwm_db: dict, node_name, gc_dict) -> dict:
    """
    Takes a sequence and a PWM database and creates a dictionary containing the lowest energy for each TF

    Args:
        seq(str): The DNA sequence
        pwm_db(dict): A dictionary with TF strings for keys and PWM dictionaries for values

    Returns:
        cns_profile(dict): A dictionary with TF strings for keys and lowest binding energy for values

    """
    # If node name is not species, pass None
    if is_node_name(node_name):
        node_name = None

    # Initialize CNS profile dictionary
    cns_profile = dict()
    # iterate through PWM_DB
    for TF, pwm in pwm_db.items():
        # get lowest binding energy
        energy, position, bind_seq = getBindingEnergy(seq, pwm, gc_dict ,node_name)
        # add to CNS profile
        cns_profile[TF] = {}
        cns_profile[TF]['Energy'] = energy
        cns_profile[TF]['Bind_Pos'] = position
        cns_profile[TF]['Bind_Seq'] = bind_seq
    return cns_profile

def count_initial_dashes(seq):
    count = 0
    for i in range(len(seq)):
        if seq[i] == '-':
            count += 1
        else: 
            return count
        
def align_start_match(seq1, pos1, seq2, pos2):
    """
    Determines if two sequences start at the same relative position in alignment.

    Args:
        seq1 (str): First sequence including possible 'N's at the beginning.
        pos1 (int): Match position of the first sequence (can be negative).
        seq2 (str): Second sequence including possible 'N's at the beginning.
        pos2 (int): Match position of the second sequence (can be negative).

    Returns:
        bool: True if the sequences start at the same relative position, False otherwise.
    """
    extension1, extension2 = float('inf'), float('inf')
    if pos1 <= 0:
        idx1 = abs(pos1)
        extension1 = idx1 - count_initial_dashes(seq1)  
    if pos2 <= 0:
        idx2 = abs(pos2)
        extension2 = idx2 - count_initial_dashes(seq2)
    
    extension = min(extension1,extension2)
    seq1 = extension * '-' + seq1
    seq2 = extension * '-' + seq2    

    # Calculate the actual starting index in the alignment for each sequence
    actual_start1 = pos1 + seq1.index(seq1.lstrip('-'))
    actual_start2 = pos2 + seq2.index(seq2.lstrip('-'))

    # Compare the starting indices
    return actual_start1 == actual_start2

def check_same_position(seq1, idx1, seq2, idx2):
    """
    Check if indices idx1 in seq1 and idx2 in seq2 correspond to the same position in aligned sequences.

    Parameters:
        seq1 (str): First aligned sequence.
        idx1 (int): Index in the first sequence.
        seq2 (str): Second aligned sequence.
        idx2 (int): Index in the second sequence.

    Returns:
        bool: True if indices correspond to the same alignment position, False otherwise.
    """
    # Ensure sequences are the same length
    if len(seq1) != len(seq2):
        raise ValueError("Aligned sequences must be of the same length.")
        
    # Handle negative indices by wrapping around
    if idx1 <=0 or idx2 <=0:
        return align_start_match(seq1, idx1, seq2, idx2)
        
    if '-' in seq1:
        idx1 += seq1.strip('-').count('-')
    if '-' in seq2:
        idx2 += seq2.strip('-').count('-')
    

    position1 = sum(1 for i in range(idx2) if seq1[i] != '-')
    position2 = sum(1 for i in range(idx1) if seq2[i] != '-')

    # Check if the alignment positions are the same
    return position1 == position2



def getEvents(parent_node, child_node, pwm_db, sequences, gc_dict):
    """
    This function takes two sequences, generates TF binding profiles for both using getCNSbindProfile, 
    calculates the change in energy for each TFBS and classifies the events that happened from CNS_A to CNS_B.

    Args: 
        seq_A(str): Sequence of prior CNS
        seq_B(str): Sequence of latter CNS
        pwm_db(dict): A dictionary containing a TF names as keys and pwms as values

    Returns:
        res(dict): result dictionary containing for each TF - prior energy, current energy, difference and classification
    """
    parent_name, parent_seq = unload_node(parent_node, sequences)
    child_name, child_seq = unload_node(child_node, sequences)
    
    # Calculate energy profiles
    parent_profile = getCNSbindProfile(parent_seq, pwm_db, parent_name, gc_dict)
    child_profile = getCNSbindProfile(child_seq, pwm_db, child_name, gc_dict)

    # Initialize result dictionary
    res = {}
    
    # Iterate through TFs and get results
    for tf in pwm_db.keys():
        res[tf] = {}
        res[tf]['Parent_Energy'] = parent_profile[tf]['Energy']
        res[tf]['Parent_Position'] = parent_profile[tf]['Bind_Pos']
        res[tf]['Parent_Bind_Seq'] = parent_profile[tf]['Bind_Seq']
        res[tf]['Child_Energy'] = child_profile[tf]['Energy']
        res[tf]['Child_Position'] = child_profile[tf]['Bind_Pos']
        res[tf]['Child_Bind_Seq'] = child_profile[tf]['Bind_Seq']
        res[tf]['Difference'] = child_profile[tf]['Energy'] -  parent_profile[tf]['Energy']
        res[tf]['Same_Pos'] = check_same_position(sequences[parent_name], res[tf]['Parent_Position'],
                                                  sequences[child_name], res[tf]['Child_Position'])
    return res

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

def getEventsHistory(node, cns_id, pwm_db, thres_db, sequences, output_file, gc_dict):
    """
    a recursive function that traverses a tree of reconstructed CNSs, and logs the TFBS events for each node using getEvents. 

    Base case:
    If node is a leaf, return None.

    Else, get events for left node and right node, log the events to a csv and calls the function in recursion for left and right node.


    Parameters:
        tree (TreeNode): The root node of the binary tree.
        cns_id (int): String with CNS ID.
        pwm_db(dict): Dictionary containing TF names as keys and PWMs as values

    Returns:
        None
    """
    if len(node.clades) == 0:
        return
    for child in node.clades:
        getEventsHistory(child, cns_id, pwm_db, thres_db, sequences, output_file, gc_dict)
        # Retrieving events between current node and its child
        res = getEvents(node, child, pwm_db, sequences, gc_dict)
        with open(output_file,'a') as out:
            writer = csv.writer(out)
            for tf in pwm_db.keys():
                parent_energy, child_energy = res[tf]['Parent_Energy'], res[tf]['Child_Energy']
                parent_binding, child_binding = classify_binding_energy(parent_energy, tf ,thres_db), classify_binding_energy(child_energy, tf ,thres_db) 
                row = [cns_id, tf, node.name, res[tf]['Parent_Position'],res[tf]['Parent_Bind_Seq'], round(parent_energy, 6), 
                    child.name ,res[tf]['Child_Position'],res[tf]['Child_Bind_Seq'], round(child_energy, 6),
                    round(res[tf]['Difference'],6), parent_binding, child_binding, res[tf]['Same_Pos']]
                writer.writerow(row)   

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run event history computation using reconstructed sequences.")

    # Define optional flagged arguments with defaults
    parser.add_argument(
        "--cns_id", 
        type=str, 
        required=True, 
        help="CNS ID to process (required)"
    )
    parser.add_argument(
        "--data_folder", 
        type=str, 
        default=DATA_FOLDER, 
        help="Data folder path (default: '../data/')"
    )
    parser.add_argument(
        "--output_folder", 
        type=str, 
        default=OUTPUT_FOLDER, 
        help="Output folder path (default: '../output/')"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call the main_model function with parsed arguments
    output_file = initialize_output_file(args.cns_id, args.output_folder + 'events/', '.events.csv')
    gc_dict = create_gc_dict(data_folder=args.data_folder)
    pwm_db = createPWMdb(data_folder=args.data_folder)
    thres_db = importThresDB(data_folder=args.data_folder)
    tree, sequences = import_tree_and_sequences(args.cns_id, args.data_folder)
    getEventsHistory(tree.root, args.cns_id, pwm_db, thres_db, sequences, output_file, gc_dict=gc_dict)