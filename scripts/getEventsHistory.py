import csv
import os
import re
import argparse
from Bio import Phylo
from config import OUTPUT_FOLDER
from reconstruct import reconstructed_fasta_to_dict
from getBindingEnergy import getBindingEnergy, classify_binding_energy, createPWMdb, create_gc_dict, importThresDB, filter_pwm_db



def import_tree_and_sequences(cns_id, data_folder=OUTPUT_FOLDER):
    """Imports a phylogenetic tree and reconstructed sequences for a given CNS ID.

    This function reads a Newick tree file and a FASTA file containing
    reconstructed ancestral sequences, both associated with a specific CNS ID.
    It assumes a specific directory structure (see Notes).

    Args:
        cns_id (str): The CNS ID (Conserved Non-coding Sequence ID).
        data_folder (str, optional): The path to the main data directory.
            Defaults to DATA_FOLDER (defined elsewhere in the script/configuration).

    Returns:
        tuple: A tuple containing:
            - tree (Bio.Phylo.BaseTree.Tree): A Biopython tree object representing the phylogeny.
            - sequences (dict): A dictionary where keys are node names and values are
            sequences 

    Raises:
        FileNotFoundError: If the tree file or sequence file for the given
            CNS ID does not exist.

    Example:
        >>> tree, sequences = import_tree_and_sequences("CNS123")
        >>> print(tree.root.name)  # Accessing the root of the tree
        'Anc0'
        >>> print(sequences["Anc0"]) # Accessing a sequence
        'ATGC...'

    Notes:
        - Assumes the following directory structure within `data_folder`:
            - `data_folder/sequences/`: Contains FASTA files with reconstructed sequences.
            - `data_folder/trees/`: Contains Newick tree files named as `{cns_id}.trimmed.annotated.tree`.
        -  The tree files are expected to be trimmed and annotated.
    """
    sequence_path = os.path.join(data_folder, 'sequences')
    tree_path = os.path.join(data_folder, 'trees', f'{cns_id}.trimmed.annotated.tree')
    
    #basic error handling:
    if not os.path.exists(tree_path):
        raise FileNotFoundError(f"Tree file not found: {tree_path}")
    if not os.path.exists(sequence_path):
        raise FileNotFoundError(f"Sequence directory not found: {sequence_path}")
    
    # Import reconstructed sequences and trimmed tree
    sequences = reconstructed_fasta_to_dict(cns_id, sequence_path)
    tree = Phylo.read(tree_path, format='newick')
    return tree, sequences


def is_node_name(string):
    """
    Checks if a given string is a node name conforming to the pattern "N#", 
    where '#' represents one or more digits.

    Args:
        string (str): The string to be checked.

    Returns:
        bool: True if the string matches the node name pattern, False otherwise.

    Example:
        >>> is_node_name("N123")
        True
        >>> is_node_name("N5")
        True
        >>> is_node_name("Node1")
        False
        >>> is_node_name("N")
        False
        >>> is_node_name("123N")
        False
    """
    if not isinstance(string, str):
        return False
    return re.match(r'^N\d+$', string) is not None
    
def unload_node(node, sequences):
    """
    Extracts the node name and its corresponding sequence from a sequences dictionary.

    Args:
        node (object): A node object with a 'name' attribute.
        sequences (dict): A dictionary where keys are node names (strings) and 
                          values are corresponding sequences.

    Returns:
        tuple: A tuple containing the node's name (str) and its sequence (any).

    Raises:
        KeyError: If the node's name is not found in the sequences dictionary.

    Example:
        >>> class Node:
        ...     def __init__(self, name):
        ...         self.name = name
        >>> node = Node("N1")
        >>> sequences = {"N1": "ATGC", "N2": "CGTA"}
        >>> unload_node(node, sequences)
        ('N1', 'ATGC')
    """
    if not hasattr(node, 'name'):
      raise ValueError("Node object must have a 'name' attribute.")
    return node.name, sequences[node.name]

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
    if is_node_name(node_name) or node_name not in gc_dict.keys():
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
        res[tf]['Same_Pos'] = int(check_same_position(sequences[parent_name], res[tf]['Parent_Position'],
                                                  sequences[child_name], res[tf]['Child_Position']))
    return res

def getEventsHistory(node, sequences, cns_id, pwm_db, thres_db, gc_dict, output_file):
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
        getEventsHistory(child, sequences, cns_id, pwm_db, thres_db, gc_dict, output_file)
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

def main(cns_id, output_folder=OUTPUT_FOLDER, tf_names = None):
    """
    Executes the main pipeline for event history reconstruction.

    This function orchestrates the process of reconstructing the evolutionary TF binding event 
    history for a given CNS (Conserved Non-coding Sequence) identified by `cns_id`. Uses all PWMs in the data folder.
    It performs the following steps:

    1.  Sets up the output directory structure.
    2.  Initializes databases for PWMs (Position Weight Matrices), thresholds, and GC content.
    3.  Filters the PWM database based on user-specified transcription factors.
    4.  Imports the phylogenetic tree and corresponding sequence data for the given CNS.
    5.  Triggers the event history reconstruction process, storing the results in a CSV file.

    Args:
        cns_id (str): The identifier for the CNS being analyzed.
        output_folder (str, optional): The base directory for output files. 
                                        Defaults to the global OUTPUT_FOLDER.
        tf_names (list, optional): A list of transcription factor names to use. 
                                     If None, all TFs in the database are used.

    Outputs:
        A CSV file named '{cns_id}.events.csv' is created in the 'events' subdirectory 
        of the specified output folder, containing the reconstructed event history.

    Example:
        >>> main("CNS123", "results", ["TF1", "TF2"])
        # Creates 'results/events/CNS123.events.csv' using only TF1 and TF2.
    """
    events_folder = os.path.join(output_folder, 'events')
    os.makedirs(events_folder, exist_ok=True)
    output_file = os.path.join(events_folder, f'{cns_id}.events.csv')

    # Initialize (or clear) the output file
    open(output_file, 'w').close()


    pwm_db, thres_db, gc_dict = createPWMdb(), importThresDB(), create_gc_dict()

    if tf_names:
        pwm_db = filter_pwm_db(pwm_db, tf_names)

    tree, sequences = import_tree_and_sequences(cns_id, output_folder)
    getEventsHistory(tree.root, sequences, cns_id, pwm_db, thres_db, gc_dict, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    Reconstructs the evolutionary binding event history for a given Conserved Non-coding Sequence (CNS).

    This script takes a CNS identifier as input, retrieves the corresponding phylogenetic 
    tree and sequence data, and applies a TF binding energy model to predict evolutionary events that occurred along 
    the branches of the tree. The results are stored in a CSV file.
    
    Example Usage:
        python script.py CNS123 -o results -t TF1 TF2 TF3
        python script.py CNS456
    """)

    parser.add_argument("cns_id", help="The identifier for the CNS to analyze.")
    parser.add_argument("-t", "--tfs", nargs='+', help="List of transcription factor names (space-separated) to use. If not provided, all TFs are used.")

    args = parser.parse_args()

    try:
        main(args.cns_id, OUTPUT_FOLDER, args.tfs)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)