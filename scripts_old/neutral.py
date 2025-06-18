import sys
import os
import random
import numpy as np
from Bio import Phylo
import pandas as pd
from model import *
from utils import *
import argparse

N_SIMULATIONS = 1000
RANDOM_SEED = 42
INCONCLUSIVE_NT = ['R', 'Y', 'S', 'W', 'K', 'M', 'B', 'D', 'H', 'V', 'N', 'X','-']
MIN_EVENTS = 2
BINDING_DICT = {'Strong': 'Bind',
                'Medium': 'Bind',
                'Weak' : 'Weak',
                'None' : 'None'}
BIND_TO_BIND = ['Bind_to_Bind', 'Bind_to_Weak', 
                'Weak_to_Bind', 'Weak_to_Weak']
BIND_CLASSES = ['Bind', 'Weak','None']
EVENTS = [bind_a +'_to_' + bind_b for bind_a in BIND_CLASSES for bind_b in BIND_CLASSES]
OUTPUT_FILE = 'Angiosperms.V10.1.simulation.csv'

def get_prob_mat(data_folder= DATA_FOLDER):
    return pd.read_csv(data_folder+'probability_matrix.csv', index_col= 0)

def get_same_pos_dict(cns_id, output_folder):
    import csv
    from collections import defaultdict
    
    def read_bool(val):
        if val == "False" or not val:
            return False
        return True

    def convert_to_standard_dict(d):
        if isinstance(d, defaultdict):
            return {k: convert_to_standard_dict(v) for k, v in d.items()}
        return d

    events_file = output_folder + cns_id + '.events.csv'
    same_pos_dict = defaultdict(lambda: defaultdict(dict))

    with open(events_file, 'r') as events:
        reader = csv.reader(events)
        for row in reader:
            tf, parent, child, same_pos = row[1], row[2], row[6], row[-1]
            same_pos_dict[tf][parent][child] = read_bool(same_pos)

    return convert_to_standard_dict(same_pos_dict)

def get_binding_tfs(cns_id, output_folder, min_bind = 2):
    df = pd.read_csv(output_folder + cns_id + '.events.csv',header= None,
                    names = ['CNS_id', 'TF',
                            'Parent_name','Parent_pos','Parent_seq',
                            'Parent_energy','Child_name','Child_pos',
                            'Child_seq','Child_energy','Difference',
                            'Parent_bind','Child_bind','Same_pos'])
    result = df.pivot_table(index='TF', columns='Child_bind', aggfunc='size', fill_value=0)
    for col in ['Strong', 'Medium','Weak']:
        if col not in result.columns:
            result[col] = 0
    total = result.Strong + result.Medium + result.Weak
    return list(total[total >= min_bind].index) 

def compute_tree_mutation_matrix(tree, sequences, ignore_chars = INCONCLUSIVE_NT):
    """
    Traverses a phylogenetic tree and computes a mutation matrix based on
    sequence differences between parent and child nodes.
    
    Parameters:
        tree (Bio.Phylo.BaseTree.Tree): The phylogenetic tree.
        sequences (dict): A dictionary where keys are node names and values are sequences.
        
    Returns:
        pd.DataFrame: The mutation matrix as a pandas DataFrame.
    """
    # Define nucleotides and initialize the mutation matrix
    nucleotides = ['A', 'C', 'G', 'T']
    mutation_matrix = pd.DataFrame(0, index=nucleotides, columns=nucleotides)
    
    # Define characters to ignore
    
    def traverse(node):
        """
        Recursive function to traverse the tree and update the mutation matrix.
        """
        if not node.clades:  # Base case: no children
            return
        
        # Get the sequence of the current node
        parent_sequence = sequences.get(node.name, None)
        
        for child in node.clades:
            # Get the sequence of the child node
            child_sequence = sequences.get(child.name, None)
            
            # If both parent and child have sequences, compare them
            if parent_sequence and child_sequence:
                for nt1, nt2 in zip(parent_sequence, child_sequence):
                    if nt1 in ignore_chars or nt2 in ignore_chars:
                        continue  # Skip invalid characters
                    if nt1 != nt2:  # Only count mutations
                        mutation_matrix.loc[nt1, nt2] += 1
            
            # Recursively traverse the child
            traverse(child)
    
    # Start traversal from the root
    traverse(tree.root)
    
    return mutation_matrix

def divide_by_rows(df):
    return df.div(df.sum(axis = 1), axis = 0)

def classify_seq_binding(seq: str, tf: str, pwm_db: dict, thres_db: dict) -> str:
    """
    Classifies the binding strength of a given DNA sequence for a specified transcription factor (TF).
    
    Args:
        seq (str): The DNA sequence to classify.
        tf (str): The transcription factor name.
        pwm_db (dict): A dictionary where keys are TF names and values are PWMs (Position Weight Matrices).
        thres_db (dict): A dictionary where keys are TF names and values are threshold energy scores.
        
    Returns:
        str: The binding classification as a string (e.g., "Strong", "Medium", "Weak", or "None").
    """
    energy, pos, bind_seq = getBindingEnergy(seq, pwm_db[tf])
    binding = classify_binding_energy(energy, tf, thres_db)
    return binding, pos

def count_substitutions(s1: str, s2:str, inconclusive = INCONCLUSIVE_NT):
    """
    Count the number of character substitutions needed to transform s1 into s2.
    Does not count substitutions if either character is '-'.
    Assumes s1 and s2 are of the same length.

    Parameters:
        s1 (str): The first string.
        s2 (str): The second string.

    Returns:
        int: The number of substitutions required.

    Raises:
        ValueError: If the strings are not of the same length.
    """
    if len(s1) != len(s2):
        raise ValueError("Strings must be of the same length to count substitutions.")
    
    # Count mismatched characters
    inconclusive = ['-','X','N']
    substitutions = sum(1 for a, b in zip(s1, s2) if a != b and a not in inconclusive and b not in inconclusive)
    return substitutions

def mutate_sequence(sequence, prob_matrix, num_mutations):
    """
    Mutates a sequence based on a probability matrix and a specified number of mutations.
    
    Parameters:
        sequence (str): The original sequence to mutate.
        prob_matrix (pd.DataFrame): The mutation probability matrix as a pandas DataFrame.
                                    Rows represent original bases, columns represent mutated bases.
        num_mutations (int): The number of mutations to introduce.
    
    Returns:
        str: The mutated sequence.
    """
    # Ensure the number of mutations is within bounds
    if num_mutations > len(sequence):
        raise ValueError("Number of mutations cannot exceed the length of the sequence.")
    
    # Generate random non-repeating indices
    mutation_indices = random.sample(range(len(sequence)), num_mutations)
    
    # Convert the sequence to a list (strings are immutable)
    mutated_sequence = list(sequence)
    
    for index in mutation_indices:
        original_base = mutated_sequence[index]
        
        # Get the mutation probabilities for the original base
        if original_base not in prob_matrix.index:
            raise ValueError(f"Base '{original_base}' not found in the probability matrix.")
        
        probabilities = prob_matrix.loc[original_base]
        # Generate a new base based on the probabilities
        mutated_base = np.random.choice(probabilities.index, p=probabilities.values)
        
        # Assign the mutated base
        mutated_sequence[index] = mutated_base
    
    # Join the list back into a string
    return ''.join(mutated_sequence)

def initialize_event_counts():
    return {event: 0 for event in EVENTS}

def is_bind_to_bind(event_count):
    for event in BIND_TO_BIND:
        if event_count[event] != 0:
            return True
    return False

def binding_to_symbol(seq_binding):
    return BINDING_DICT[seq_binding]

def get_event_symbol(seq_binding, mut_binding):
    seq_symbol, mut_symbol = binding_to_symbol(seq_binding), binding_to_symbol(mut_binding)
    return seq_symbol + '_to_' + mut_symbol

def get_same_pos_frac(same_pos, expected_events):
    if is_bind_to_bind(expected_events):
        bind_to_bind_events = sum([expected_events[event] for event in BIND_TO_BIND])
        return same_pos/bind_to_bind_events
    else:
        return 0

def update_event_dictionary(seq_binding_1, seq_2, tf,event_dictionary, pwm_db, thres_db):
    seq_binding_2, seq2_pos = classify_seq_binding(seq_2, tf, pwm_db, thres_db)
    event_symbol = get_event_symbol(seq_binding_1, seq_binding_2)
    event_dictionary[event_symbol] += 1
    return event_dictionary, event_symbol, seq2_pos 
    
def create_row(cns_id, tf, parent, child, subs, observed_events, expected_events, n_simulations, observed_same_pos ,same_pos):
    same_pos_frac = get_same_pos_frac(same_pos, expected_events)
    return [cns_id, tf, parent, child, subs] + [observed_events[event] for event in EVENTS] + [observed_same_pos] + [expected_events[event]/n_simulations for event in EVENTS] + [same_pos_frac]

def process_node(cns_id, tf, sequences, prob_mat, parent_name, child_name, pwm_db, thres_db, same_pos_dict, num_simulations):
    parent_seq, child_seq = sequences[parent_name], sequences[child_name]
    observed_events, expected_events = initialize_event_counts(), initialize_event_counts()
    subs = count_substitutions(parent_seq, child_seq)
    parent_binding, parent_bind_pos = classify_seq_binding(parent_seq, tf, pwm_db, thres_db)
    observed_events = update_event_dictionary(parent_binding, child_seq, tf, observed_events, pwm_db, thres_db)[0]
    if is_bind_to_bind(observed_events):
        observed_same_pos = int(same_pos_dict[tf][parent_name][child_name])
    else:
        observed_same_pos = 0
    same_pos = 0
    if subs != 0:
        for i in range(num_simulations):
            mut_seq = mutate_sequence(parent_seq, prob_mat, subs)
            expected_events, event_symbol, mut_bind_pos = update_event_dictionary(parent_binding, mut_seq, tf, expected_events, pwm_db, thres_db)
            if event_symbol in BIND_TO_BIND: # Check whether bind site moved
                if check_same_position(parent_seq, parent_bind_pos, mut_seq, mut_bind_pos):
                    same_pos += 1
                
    res = create_row(cns_id, tf, parent_name, child_name, subs, observed_events, expected_events, num_simulations, observed_same_pos ,same_pos)

    return res

def getEventsProbability(cns_id, node, sequences, prob_mat, n_simulations, tf, pwm_db, thres_db, same_pos_dict, output_file):
    if len(node.clades) == 0:
        return
    else:
        parent_name = node.name
        for child_clade in node.clades:
            child_name = child_clade.name
            res = process_node(cns_id, tf, sequences, prob_mat, parent_name, child_name, pwm_db, thres_db, same_pos_dict, n_simulations)                
            # Write the row to the CSV file
            with open(output_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(res)
            
            getEventsProbability(cns_id, child_clade, sequences, prob_mat, n_simulations, tf, pwm_db, thres_db, same_pos_dict, output_file)

def neutral_main(cns_id, tree, sequences, n_simulations, pwm_db, thres_db, prob_mat, output_folder, min_events = MIN_EVENTS):
    events_folder =  output_folder + 'events/'
    output_file = initialize_output_file(cns_id, output_folder+'simulations/', '.simulation.csv')
    binding_tfs = get_binding_tfs(cns_id, events_folder, min_events)
    same_pos_dict = get_same_pos_dict(cns_id, events_folder)
    for tf in binding_tfs:
        getEventsProbability(cns_id, tree.root, sequences, prob_mat, n_simulations, tf, pwm_db, thres_db, same_pos_dict, output_file)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run neutral simulations for CNS analysis.")
    parser.add_argument("--cns_id", type=str, help="CNS ID to analyze.")
    parser.add_argument("--n_simulations", type=int, default=1000, help="Number of simulations to run (default: 1000).")
    parser.add_argument("--output_folder", type=str, default="../output/", help="Path to the output folder (default: ../output/).")
    parser.add_argument("--data_folder", type=str, default="../data/", help="Path to the data folder (default: ../data/).")

    args = parser.parse_args()

    cns_id = args.cns_id
    n_simulations = args.n_simulations
    output_folder = args.output_folder
    data_folder = args.data_folder

    tree_file = f"{data_folder}trees/{cns_id}.trimmed.annotated.tree"
    pwm_db = createPWMdb()
    thres_db = importThresDB()
    prob_mat = get_prob_mat(data_folder)
    sequences = reconstructed_fasta_to_dict(cns_id, f"{data_folder}sequences/")
    tree = Phylo.read(tree_file, format="newick")

    neutral_main(cns_id, tree, sequences, n_simulations, pwm_db, thres_db, prob_mat, output_folder, min_events=MIN_EVENTS)
