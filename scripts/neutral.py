import pandas as pd
import random
import numpy as np
import os

from config import DATA_FOLDER, OUTPUT_FOLDER
from getBindingEnergy import getBindingEnergy, classify_binding_energy

BINDING_DICT = {'Strong': 'Bind',
                'Medium': 'Bind',
                'Weak' : 'Weak',
                'None' : 'None'}
BIND_TO_BIND = ['Bind_to_Bind', 'Bind_to_Weak', 
                'Weak_to_Bind', 'Weak_to_Weak']
BIND_CLASSES = ['Bind', 'Weak','None']
EVENTS = [bind_a +'_to_' + bind_b for bind_a in BIND_CLASSES for bind_b in BIND_CLASSES]

def get_binding_tfs(cns_id: str, output_folder: str = OUTPUT_FOLDER, min_bind: int = 2) -> list:
    """
    Filters model results to identify transcription factors (TFs) that bind a sequence frequently.

    This function first checks for an 'events' subdirectory within the specified
    output folder. If the directory doesn't exist, it raises a FileNotFoundError.
    
    If the directory exists, it reads a detailed events CSV file from it. It
    counts the total number of 'Strong', 'Medium', and 'Weak' binding events
    for each transcription factor and returns a list of TFs that have a total
    binding count equal to or greater than the specified minimum.

    Args:
        cns_id (str): The identifier for the conserved non-coding sequence (CNS),
                      used to construct the input filename (e.g., 'cns_1').
        output_folder (str): The path to the main directory containing the 'events/'
                             subdirectory.
        min_bind (int, optional): The minimum number of total binding events
                                  required for a TF to be returned. Defaults to 2.

    Returns:
        list: A list of TF names (str) that meet the minimum binding event threshold.

    Raises:
        FileNotFoundError: If the 'events/' subdirectory does not exist within the
                           output_folder.
    """
    # Define the path to the events directory
    events_dir = os.path.join(output_folder, 'events/')

    # Check if the events directory exists
    if not os.path.isdir(events_dir):
        raise FileNotFoundError(
            f"The events folder at '{events_dir}' does not exist. "
            "Please run the getEventsHistory.py script first."
        )
    
    # Define column names for the event file, which is expected to have no header
    col_names = [
        'CNS_id', 'TF', 'Parent_name', 'Parent_pos', 'Parent_seq',
        'Parent_energy', 'Child_name', 'Child_pos', 'Child_seq',
        'Child_energy', 'Difference', 'Parent_bind', 'Child_bind', 'Same_pos'
    ]

    # Construct the full file path to the specific events file
    file_path = os.path.join(events_dir, f"{cns_id}.events.csv")

    # Read the specific event data for the given CNS ID
    df = pd.read_csv(
        file_path,
        header=None,
        names=col_names
    )

    # Create a pivot table to count binding events ('Strong', 'Medium', 'Weak') for each TF
    result = df.pivot_table(index='TF', columns='Child_bind', aggfunc='size', fill_value=0)

    # Ensure all binding strength columns exist to prevent errors during summation
    for col in ['Strong', 'Medium', 'Weak']:
        if col not in result.columns:
            result[col] = 0

    # Calculate the total binding events for each TF
    total_events = result['Strong'] + result['Medium'] + result['Weak']

    # Filter for TFs that meet the minimum binding threshold and return their names as a list
    binding_tfs = total_events[total_events >= min_bind]
    
    return list(binding_tfs.index)


def classify_seq_binding(seq: str, tf: str, pwm_db: dict, thres_db: dict, gc_dict: dict, species = None) -> str:
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
    energy, pos, bind_seq = getBindingEnergy(seq, pwm_db[tf], gc_dict, species)
    binding = classify_binding_energy(energy, tf, thres_db)
    return binding, pos

def mutate_sequence(sequence: str, prob_matrix: pd.DataFrame, num_mutations: int):
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


def count_substitutions(s1: str, s2:str):
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

def initialize_event_counts():
  """Initializes a dictionary to store counts for each event type.

  This function creates a new dictionary where each key corresponds to an event
  from the global `EVENTS` constant. All event counts are initialized to 0.

  Returns:
      dict: A dictionary with event names (str) as keys and their counts (int)
            as values, with all counts set to 0.
            Example: {'Bind_to_Bind': 0, 'Bind_to_Weak': 0, 'Bind_to_None': 0}
  """
  return {event: 0 for event in EVENTS}