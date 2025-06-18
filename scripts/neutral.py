import csv
import pandas as pd
import random
import numpy as np
import os
import argparse
from Bio import Phylo

from config import DATA_FOLDER, OUTPUT_FOLDER
from getBindingEnergy import getBindingEnergy, classify_binding_energy
from getEventsHistory import check_same_position, is_node_name
from model_utils import *

# Converts energy model results to binding classifications in the neutral model
# Main difference is Strong and Medium are consolidated into high likelihood of binding
BINDING_DICT = {'Strong': 'Bind',
                'Medium': 'Bind',
                'Weak' : 'Weak',
                'None' : 'None'}
BIND_TO_BIND = ['Bind_to_Bind', 'Bind_to_Weak', 
                'Weak_to_Bind', 'Weak_to_Weak']
BIND_CLASSES = ['Bind', 'Weak','None']
EVENTS = [bind_a +'_to_' + bind_b for bind_a in BIND_CLASSES for bind_b in BIND_CLASSES]
N_SIMULATIONS = 1000
MIN_EVENTS = 2

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

def is_bind_to_bind(event_count):
    """
    Tests whether a reservation of binding event has occurred based on event counts.

    This function iterates through a predefined list of event strings that represent
    a "reservation of binding" (e.g. Strong_to_Strong, Strong_to_Weak etc.). It checks if any of these specific events have a
    non-zero count in the provided event_count dictionary.

    Args:
        event_count (dict): A dictionary where keys are event names (str) and
                            values are their corresponding counts (int).

    Returns:
        bool: True if any of the "reservation of binding" events have a count
              greater than zero, False otherwise.
    """
    for event in BIND_TO_BIND:
        if event_count[event] != 0:
            return True
    return False

def binding_to_symbol(seq_binding):
    """
    Converts a sequence binding identifier to its corresponding symbol.

    This function acts as a simple lookup, translating a given binding
    identifier (string) into a predefined symbol by looking it up in a
    global dictionary named `BINDING_DICT`.

    Args:
        seq_binding (str): The binding identifier for a sequence.

    Returns:
        str: The symbol corresponding to the provided binding identifier.
    """
    return BINDING_DICT[seq_binding]

def get_event_symbol(seq_binding, mut_binding):
    """
    Generates a composite event symbol from sequence and mutation bindings.

    This function creates a standardized event symbol string by first converting
    both a sequence binding and a mutation binding into their respective symbols
    using the `binding_to_symbol` function. It then concatenates these symbols
    with "_to_" to form a descriptive event name.

    Args:
        seq_binding (str): The binding identifier for the sequence.
        mut_binding (str): The binding identifier for the mutation.

    Returns:
        str: A formatted string representing the combined event symbol,
             in the format 'sequenceSymbol_to_mutationSymbol'.

    Example:
        >>> get_event_symbol('Strong', 'Medium')
        'Bind_to_Bind'
    """
    seq_symbol, mut_symbol = binding_to_symbol(seq_binding), binding_to_symbol(mut_binding)
    return seq_symbol + '_to_' + mut_symbol

def get_same_pos_frac(same_pos, expected_events):
    """
    Calculates the fraction of events where binding was reserved in the same position to all bind_to_bind events.
    Reflects on how many cases the binding was retained due to a redundant binding site or a change the moved the 
    binding site. 

    Args:
        same_pos (int or float): The number of events that occurred at the same position.
        expected_events (dict): A dictionary where keys are event names (str)
                                and values are their counts (int).

    Returns:
        float: The calculated fraction of same-position events to total
               bind-to-bind events. Returns 0 if there are no bind-to-bind
               events or if their total count is zero, preventing a
               division-by-zero error.
    """
    if is_bind_to_bind(expected_events):
        bind_to_bind_events = sum([expected_events[event] for event in BIND_TO_BIND])
        return same_pos/bind_to_bind_events
    else:
        return 0
    
def update_event_dictionary(seq_binding_1, seq_2, tf, event_dictionary, pwm_db, thres_db, gc_dict, species = None):
    """
    Classifies the change in binding between two sequences, and returns and updated event count dictionary.

    This function orchestrates several steps to track changes in binding events.
    First, it classifies the binding nature of a given sequence (`seq_2`) with respect
    to a transcription factor (`tf`) using Position Weight Matrices (PWMs) and
    predefined thresholds.

    Next, it generates a composite event symbol that represents the transition
    from an initial binding state (`seq_binding_1`) to the newly determined state of `seq_2`.

    Finally, it increments the count for this specific event in the provided
    `event_dictionary` and returns the updated dictionary along with the symbol
    of the new event and the position of the binding in the second sequence.

    Args:
        seq_binding_1 (str): A string identifier for the binding state of the
                             initial sequence (e.g., 'strong_binder', 'no_binder').
        seq_2 (str): The DNA/protein sequence to be classified.
        tf (str): The identifier for the transcription factor being analyzed.
        event_dictionary (dict): A dictionary where keys are event symbols (str)
                                 and values are their counts (int). This dictionary
                                 is updated by the function.
        pwm_db (dict): A database or dictionary containing Position Weight Matrix
                       (PWM) data required by `classify_seq_binding`.
        thres_db (dict): A database or dictionary containing binding threshold values
                         required by `classify_seq_binding`.
        gc_dict (dict): a dictionary of GC fractions by species or a mean GC fraction for extinct species.
        species (string): a species identifier for species that exist in the database.

    Returns:
        tuple: A tuple containing three elements:
            - event_dictionary (dict): The updated dictionary with the new event count.
            - event_symbol (str): The composite symbol for the observed event
              (e.g., 'Bind_to_None').
            - seq2_pos (int or None): The position of the binding site found in `seq_2`.

    Note:
        This function depends on two other functions being available in the scope:
        - `classify_seq_binding(sequence, tf, pwm_db, thres_db)`: Which must return the
          binding classification and position for a given sequence.
        - `get_event_symbol(binding_1, binding_2)`: Which must return a standardized
          string representing the transition between two binding states.
    """
    # Classify the second sequence to determine its binding state and position.
    seq_binding_2, seq2_pos = classify_seq_binding(seq_2, tf, pwm_db, thres_db, gc_dict, species)

    # Generate the event symbol for the transition from the first to the second binding state.
    event_symbol = get_event_symbol(seq_binding_1, seq_binding_2)

    # Increment the count for this specific event in the dictionary.
    event_dictionary[event_symbol] += 1

    # Return the updated dictionary and the details of the new event.
    return event_dictionary, event_symbol, seq2_pos

def create_row(cns_id, tf, parent, child, subs, observed_events, expected_events, n_simulations, observed_same_pos, same_pos):
    """
    Creates a summary data row for a link between two nodes in a phylogenetic tree.

    This function aggregates various observed and simulated data points related to
    binding events for a specific transcription factor (tf) along a single
    evolutionary branch (from parent to child node). It formats this information
    into a single list, which can be easily appended to a results table (e.g., a
    CSV file or a pandas DataFrame).

    The function calculates the fraction of "same position" binding events from
    simulations and combines it with identifiers, substitution counts, observed
    event counts, and averaged expected event counts into one comprehensive record.

    Args:
        cns_id (str or int): An identifier for the conserved non-coding sequence (CNS)
                             under analysis.
        tf (str): The identifier for the transcription factor being studied.
        parent (str): The identifier for the parent node in the phylogenetic tree.
        child (str): The identifier for the child node in the phylogenetic tree.
        subs (int): The number of nucleotide/amino acid substitutions that
                    occurred between the parent and child sequences.
        observed_events (dict): A dictionary mapping event symbols (str) to their
                                observed counts (int) on this branch.
        expected_events (dict): A dictionary mapping event symbols (str) to their
                                total counts (int) from all simulations.
        n_simulations (int): The total number of simulations performed to generate
                             the `expected_events`.
        observed_same_pos (int): The number of times a binding event was observed
                                 at the exact same position in both parent and child.
        same_pos (int): The total number of times a binding event occurred at the
                        same position across all simulations.

    Returns:
        list: A flat list representing a single row of data, structured as follows:
              - [cns_id, tf, parent, child, subs]
              - List of observed event counts, ordered by the `EVENTS` constant.
              - [observed_same_pos]
              - List of average expected event counts (normalized by `n_simulations`).
              - [fraction of same-position binding from simulations].
    """
    same_pos_frac = get_same_pos_frac(same_pos, expected_events)
    return ([cns_id, tf, parent, child, subs] + # Identifier of row 
            [observed_events[event] for event in EVENTS] +  # Observed event results
            [observed_same_pos] + # Whether event is a same position event
            [expected_events[event]/n_simulations for event in EVENTS] + # Expected event result normalized by number of simulations
            [same_pos_frac]) # Fraction of simulated events that were same position events
    
def analyze_observed_branch(parent_seq, child_seq, tf, pwm_db, thres_db, gc_dict, 
                            parent_species, child_species, same_pos_lookup):
    """
    Analyzes the observed evolutionary change between a parent and child sequence.

    Args:
        parent_seq (str): The sequence of the parent node.
        child_seq (str): The sequence of the child node.
        tf (str): The transcription factor being studied.
        pwm_db (dict): A database of Position Weight Matrices.
        thres_db (dict): A database of binding score thresholds.
        gc_dict (dict): A database of GC fractions by species.
        parent_species (str): a species identifier for GC fraction. If ancestral node, 
        is set by default to None and uses mean GC fraction.
        child_species (str): a species identifier for GC fraction. If ancestral node, 
        is set by default to None and uses mean GC fraction.
        same_pos_lookup (bool): A pre-calculated boolean indicating if the observed
                                binding site was conserved.


    Returns:
        tuple: A tuple containing:
            - observed_events (dict): A dictionary with the count for the observed event.
            - observed_same_pos (int): 1 if the binding site was conserved, 0 otherwise.
            - parent_binding (str): The binding classification of the parent sequence.
            - parent_bind_pos (int): The binding position in the parent sequence.
    """
    observed_events = initialize_event_counts()
    parent_binding, parent_bind_pos = classify_seq_binding(parent_seq, tf, pwm_db, thres_db, gc_dict, parent_species)
    
    # Classify child and update observed event counts
    observed_events = update_event_dictionary(parent_binding, child_seq, tf, observed_events, pwm_db, thres_db, child_species)[0]

    # Determine if the position was conserved for the observed event
    observed_same_pos = 0
    if is_bind_to_bind(observed_events) and same_pos_lookup:
        observed_same_pos = 1
        
    return observed_events, observed_same_pos, parent_binding, parent_bind_pos


def run_simulation_analysis(parent_seq, parent_binding, parent_bind_pos, subs, prob_mat, 
                            tf, pwm_db, thres_db, gc_dict, child_species, num_simulations):
    """
    Runs Monte Carlo simulations to generate a distribution of expected events.

    Args:
        parent_seq (str): The sequence of the parent node.
        parent_binding (str): The binding classification of the parent sequence.
        parent_bind_pos (int): The binding position in the parent sequence.
        subs (int): The number of substitutions to introduce in each simulation.
        prob_mat (dict): The substitution probability matrix.
        tf (str): The transcription factor being studied.
        pwm_db (dict): A database of Position Weight Matrices.
        thres_db (dict): A database of binding score thresholds.
        gc_dict (dict): A database of GC fractions by species.
        child_species (str): a species identifier for GC fraction. If ancestral node, 
        is set by default to None and uses mean GC fraction.
        num_simulations (int): The number of simulations to run.

    Returns:
        tuple: A tuple containing:
            - expected_events (dict): A dictionary of total event counts from all simulations.
            - simulated_same_pos (int): The number of simulations where a bind-to-bind
                                      event occurred at the same position.
    """
    expected_events = initialize_event_counts()
    simulated_same_pos = 0

    if subs == 0:
        return expected_events, simulated_same_pos

    for _ in range(num_simulations):
        mut_seq = mutate_sequence(parent_seq, prob_mat, subs)
        
        # Classify mutated sequence and update counts
        expected_events, event_symbol, mut_bind_pos = update_event_dictionary(
            parent_binding, mut_seq, tf, expected_events, pwm_db, thres_db, gc_dict, child_species
        )
        
        # If it was a bind-to-bind event, check for position conservation
        if event_symbol in BIND_TO_BIND:
            if check_same_position(parent_seq, parent_bind_pos, mut_seq, mut_bind_pos):
                simulated_same_pos += 1
                
    return expected_events, simulated_same_pos

def process_branch(cns_id, tf, sequences, prob_mat, parent_name, child_name, 
                   pwm_db, thres_db, gc_dict, same_pos_dict, num_simulations):
    """
    Analyzes a phylogenetic branch by coordinating observed and simulated analyses.

    This function orchestrates the analysis of a single evolutionary branch. It
    delegates the analysis of the real parent-child pair and the analysis of
    simulated evolutionary paths to specialized functions, then compiles the
    results into a single data row.

    Args:
        cns_id (str or int): Identifier for the conserved non-coding sequence (CNS).
        tf (str): Identifier for the transcription factor (TF) being studied.
        sequences (dict): A dictionary mapping node names to their DNA sequences.
        prob_mat (dict): The substitution probability matrix.
        parent_name (str): The identifier for the parent node of the branch.
        child_name (str): The identifier for the child node of the branch.
        pwm_db (dict): A database of Position Weight Matrices (PWMs).
        thres_db (dict): A database of binding score thresholds.
        gc_dict (dict): A database of GC fractions by species.
        same_pos_dict (dict): A pre-computed dictionary for observed same-position events.
        num_simulations (int): The number of simulations to perform.

    Returns:
        list: A single data row summarizing the complete analysis for the branch.
    """
    parent_seq, child_seq = sequences[parent_name], sequences[child_name]
    parent_species, child_species = tuple([None if not is_node_name(name) else name for name in [parent_name, child_name]])
    subs = count_substitutions(parent_seq, child_seq)

    # --- Step 1: Analyze the Observed Change ---
    same_pos_lookup = same_pos_dict[tf][parent_name][child_name]
    observed_events, observed_same_pos, parent_binding, parent_bind_pos = analyze_observed_branch(
        parent_seq, child_seq, tf, pwm_db, thres_db, gc_dict, 
        parent_species, child_species, same_pos_lookup
    )

    # --- Step 2: Run Simulations to get Expected Changes ---
    expected_events, simulated_same_pos = run_simulation_analysis(
        parent_seq, parent_binding, parent_bind_pos, subs, prob_mat, 
        tf, pwm_db, thres_db, gc_dict, child_species, num_simulations
    )
                    
    # --- Step 3: Compile the final results row ---
    res = create_row(
        cns_id, tf, parent_name, child_name, subs, observed_events, 
        expected_events, num_simulations, observed_same_pos, simulated_same_pos
    )

    return res

def getEventsProbability(cns_id, node, sequences, prob_mat, n_simulations, tf, pwm_db, thres_db, gc_dict,
                         same_pos_dict, output_file):
    """
    Recursively traverses a phylogenetic tree to analyze simulate evolutionary events in the binding affinity
    of a transcription factor to a CNS. It is used to create a background distribution of events to analyze the 
    probability of observing the given change under the assumption of neutral evolution.

    This function walks through a phylogenetic tree, starting from the given `node`.
    For each branch (parent-to-child link) it encounters, it calls the 
    `process_branch` function to perform a detailed analysis comparing observed
    evolutionary changes to simulated ones.

    The results from each branch analysis are then immediately appended as a new row
    to the specified CSV output file. The function calls itself on each child
    node to continue the traversal until all descendants have been visited.

    Args:
        cns_id (str): Identifier for the conserved non-coding sequence.
        node (Bio.Phylo.Clade): The current node object in the phylogenetic tree
                               from which to start the traversal. Must have `.name`
                               and `.clades` attributes.
        sequences (dict): A dictionary mapping node names (str) to their
                          corresponding DNA sequences (str).
        prob_mat (dict): The substitution probability matrix for simulations.
        n_simulations (int): The number of simulations to run per branch.
        tf (str): The identifier for the transcription factor being studied.
        pwm_db (dict): A database of Position Weight Matrices (PWMs).
        thres_db (dict): A database of binding score thresholds.
        gc_dict (dict): A database of GC fractions by species.
        same_pos_dict (dict): A lookup dictionary for observed positional conservation data.
        output_file (str): The file path for the output CSV file. Results will be
                           appended to this file.

    Returns:
        None: This function does not return any value. Its primary purpose is the
              side effect of writing results to a file.

    Side Effects:
        - Writes one or more data rows to the CSV file specified by `output_file`.
        - The file is opened in append mode, so existing content will be preserved.
    """
    # Base case: If the node is a leaf (has no children), stop the recursion.
    if not node.clades:
        return
    
    # Recursive step: Process each child branch from the current node.
    parent_name = node.name
    for child_clade in node.clades:
        child_name = child_clade.name
        
        # Analyze the individual branch to get a results row.
        res = process_branch(
            cns_id, tf, sequences, prob_mat, parent_name, child_name,
            pwm_db, thres_db, gc_dict, same_pos_dict, n_simulations
        )
        
        # Write the resulting row to the output file.
        try:
            with open(output_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(res)
        except IOError as e:
            print(f"Error writing to file {output_file}: {e}")
            # Decide how to handle the error: maybe stop execution or just print a warning
            return

        # Continue the traversal down the tree from the child node.
        getEventsProbability(
            cns_id, child_clade, sequences, prob_mat, n_simulations,
            tf, pwm_db, thres_db, gc_dict, same_pos_dict, output_file
        )
        
def neutral_main(cns_id, tree, sequences, n_simulations, pwm_db, thres_db,
                 gc_dict, prob_mat, output_folder, min_events):
    """
    Orchestrates the neutral evolution simulation for a given sequence.

    This function serves as the main entry point to run a neutral evolution
    analysis on a specific conserved non-coding sequence (cns_id). It prepares
    the environment by setting up directories and pre-loading necessary data.
    It then iterates through all relevant transcription factors (TFs) that show
    a minimum level of binding activity and launches a detailed, recursive
    analysis for each one across the entire phylogenetic tree.

    The workflow is as follows:
    1.  Verifies that a prerequisite 'events' folder exists.
    2.  Creates a 'simulations' output folder.
    3.  Initializes a CSV file to store the simulation results.
    4.  Identifies which TFs to analyze based on the `min_events` threshold.
    5.  Pre-loads a dictionary with positional conservation data for efficiency.
    6.  For each selected TF, it calls `getEventsProbability` to traverse the
        tree and write the analysis results for each branch to the output file.

    Args:
        cns_id (str or int): Identifier for the conserved non-coding sequence.
        tree (Bio.Phylo.BaseTree): A phylogenetic tree object (e.g., from BioPython).
        sequences (dict): A dictionary mapping node names from the tree to their
                          corresponding DNA sequences.
        n_simulations (int): The number of Monte Carlo simulations to run for each
                             branch analysis.
        pwm_db (dict): A database of Position Weight Matrices for binding analysis.
        thres_db (dict): A database of binding score thresholds.
        gc_dict (dict): A dictionary mapping node names to their GC content, likely
                        used by the mutation model.
        prob_mat (dict): The substitution probability matrix used for simulating
                         mutations under a neutral model.
        output_folder (str): The path to the main output directory where 'events'
                             and 'simulations' subfolders are located/created.
        min_events (int): The minimum number of observed events required for a
                          transcription factor to be included in the analysis.

    Returns:
        None: This function does not return a value. Its primary purpose is to
              orchestrate the analysis and save results to a file.

    Side Effects:
        - Creates a 'simulations' subdirectory inside the `output_folder`.
        - Creates a '{cns_id}.simulation.csv' file within the 'simulations' folder.
        - Writes all analysis results as rows into the created CSV file.

    Raises:
        FileNotFoundError: If the prerequisite '/events/' directory does not exist
                           within the `output_folder`.
    """
    # Define and verify the prerequisite events folder
    events_folder = os.path.join(output_folder, 'events/')
    if not os.path.isdir(events_folder):
        raise FileNotFoundError("Events folder doesn't exist. Use getEventHistory.py first.")

    # Define and create the output folder for this analysis
    simulations_folder = os.path.join(output_folder, 'simulations/')
    os.makedirs(simulations_folder, exist_ok=True)

    # Prepare the output file, likely creating it and writing a header
    output_file = initialize_output_file(cns_id, simulations_folder, '.simulation.csv')

    # --- Pre-load data before starting the main analysis loop ---

    # Get a list of TFs that meet the minimum event threshold
    binding_tfs = get_binding_tfs(cns_id, events_folder, min_events)
    # Get a lookup dictionary for observed positional conservation
    same_pos_dict = get_same_pos_dict(cns_id, events_folder)

    # --- Run the analysis for each relevant transcription factor ---
    for tf in binding_tfs:
        # Initiate the recursive traversal and analysis for the current TF
        getEventsProbability(cns_id, tree.root, sequences, prob_mat, n_simulations,
                             tf, pwm_db, thres_db, gc_dict, same_pos_dict, output_file)

        
if __name__ == '__main__':
    # --------------------------------------------------------------------------
    # Main execution block
    # --------------------------------------------------------------------------

    # --- 1. Set up Command-Line Argument Parsing ---
    # The `argparse` module allows for a flexible command-line interface,
    # enabling users to provide inputs and override default settings without
    # editing the code.
    parser = argparse.ArgumentParser(description="Run neutral simulations for CNS analysis.")
    
    # Define the arguments the script can accept
    parser.add_argument("--cns_id", type=str, required=True, help="CNS ID to analyze (e.g., 'cns_1'). This is a required argument.")
    parser.add_argument("--n_simulations", type=int, default=N_SIMULATIONS, help=f"Number of simulations to run per branch (default: {N_SIMULATIONS}).")
    parser.add_argument("--min_events", type=int, default=MIN_EVENTS, help=f"Minimal number of binding events required to process a TF (default: {MIN_EVENTS}).")
    parser.add_argument("--output_folder", type=str, default=OUTPUT_FOLDER, help=f"Path to the main output folder (default: {OUTPUT_FOLDER}).")
    parser.add_argument("--data_folder", type=str, default=DATA_FOLDER, help=f"Path to the folder containing input data like trees and sequences (default: {DATA_FOLDER}).")

    # Parse the arguments provided by the user
    args = parser.parse_args()

    # Assign parsed arguments to variables for easier access
    cns_id = args.cns_id
    n_simulations = args.n_simulations
    min_events = args.min_events
    output_folder = args.output_folder
    data_folder = args.data_folder
    
    # This section prepares all the inputs required by the main analysis function.
    # It constructs file paths and calls various helper functions to load data
    # from disk into memory.
    tree, sequences = import_tree_and_sequences(cns_id, output_folder)
    pwm_db = createPWMdb()
    thres_db = importThresDB()
    prob_mat = get_prob_mat(data_folder)

    # Run the Main Analysis Function
    neutral_main(cns_id, tree, sequences, n_simulations, pwm_db, thres_db, prob_mat, output_folder, min_events)
