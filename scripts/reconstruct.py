import os
import csv
import subprocess

import pandas as pd
from Bio import Phylo
from Bio import Align
from config import OUTPUT_FOLDER, DATA_FOLDER, FASTML_BINARY_PATH

from typing import Dict

def fasta_to_dict(file_path: str) -> Dict[str, str]:
    """Reads a FASTA file and converts it into a dictionary.

    Args:
        file_path (str): The path to the FASTA file.

    Returns:
        dict: A dictionary where keys are sequence names (headers, without the
            leading '>') and values are the corresponding sequences (as strings).
            Returns an empty dictionary if the file is empty or does not exist.

    Dependencies:
        - None (This function uses only built-in Python functionality)

    Error Handling:
        - Returns an empty dictionary if the file does not exist.
        - Does *not* explicitly handle invalid FASTA formats.  It will likely
          produce incorrect results or raise an exception in some cases of
          malformed FASTA files, but this is not guaranteed.

    Example Usage:
        >>> sequence_dict = fasta_to_dict("/path/to/my_sequences.fasta")
        >>> print(sequence_dict)
        {'Seq1': 'ATGC...', 'Seq2': 'CGTA...'}
    """
    sequences: Dict[str, str] = {}
    current_sequence_name: str = ""  # Initialize to empty string
    current_sequence: str = ""

    try:
        with open(file_path, "r") as file:
            for line in file:
                line = line.strip()
                if line.startswith(">"):
                    if current_sequence_name:
                        sequences[current_sequence_name] = current_sequence
                    current_sequence_name = line[1:]  # Remove ">"
                    current_sequence = ""
                else:
                    current_sequence += line

            # Add the last sequence after the loop finishes
            if current_sequence_name:
                sequences[current_sequence_name] = current_sequence
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
        return {}  # Return empty dict if file not found
    except Exception as e:
        print(f"An unexpected error: {e} while processing file {file_path}")
        return{}
    return sequences

def reconstructed_fasta_to_dict(cns_id: str, path: str) -> Dict[str, str]:
    """Reads a reconstructed FASTA file and converts it into a dictionary.

    Args:
        cns_id (str): The CNS ID.
        path (str): Path to the folder containing the FASTA file.

    Returns:
        dict: A dictionary where keys are sequence names (headers, without the
            leading '>') and values are the corresponding sequences (as strings).
            Returns an empty dictionary if the file does not exist or is empty.

    Dependencies:
        - None (This function relies on `fasta_to_dict`, which uses built-in Python)

    Error Handling:
        - Returns an empty dictionary if the file does not exist (handled by `fasta_to_dict`).
        - Does *not* explicitly handle invalid FASTA formats (relies on `fasta_to_dict`).

    Example Usage:
        >>> sequence_dict = reconstructed_fasta_to_dict("CNS123", "/path/to/fasta/files")
        >>> print(sequence_dict)
        {'SpeciesA': 'ATGCG...', 'SpeciesB': 'CGTA...'}
    """
    file_path = os.path.join(path, f'{cns_id}.reconstructed.fasta')
    return fasta_to_dict(file_path)


def get_ancestral_seq(cns_id, cns_file):
    """
    Retrieves the ancestral sequence for a given CNS ID from a CSV file.
    
    Args:
        cns_id (str): The CNS ID.
        cns_file (str): Path to the CNS file.
    
    Returns:
        str: The ancestral sequence if found, otherwise None.
    """
    with open(cns_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if cns_id in row:
                return row[7]  # Assuming 8th column index is 7 (0-indexed)
    return None

def compute_identity(seq1, seq2):
    """
    Computes sequence identity percentage, ignoring gaps ('-').
    
    Args:
        seq1 (str): First sequence.
        seq2 (str): Second sequence.
    
    Returns:
        float: Identity percentage (0-100).
    """
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of the same length.")
    
    match_count = sum(1 for a, b in zip(seq1, seq2) if a == b and a != '-')
    valid_positions = sum(1 for a, b in zip(seq1, seq2) if a != '-' or b != '-')
    
    if valid_positions == 0:
        return 0.0
    return (match_count / valid_positions) * 100

def realign_and_identity(ref_seq, seq):
    """
    Aligns two sequences using global alignment and computes identity.
    
    Args:
        ref_seq (str): Reference sequence.
        seq (str): Query sequence.
    
    Returns:
        float: Identity percentage.
    """
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    alignment = aligner.align(ref_seq, seq)
    aligned_a, aligned_b = alignment[0][0], alignment[0][1]
    return compute_identity(aligned_a, aligned_b)


def drop_duplicates_by_identity(df, cns_file):
    """
    Removes duplicate sequences in the same species from a DataFrame based on identity to an ancestral consensus sequence.

    This function identifies duplicate sequences 'dashed_seq' column,
    calculates their identity to an ancestral sequence retrieved from a file, and keeps only the
    sequence with the highest identity for each unique sequence. If sequences are of different lengths,
    they are realigned before identity calculation.

    Args:
        df (pd.DataFrame): A DataFrame containing sequence data. It is assumed that:
            - The first column contains a CNS ID.
            - The third column (index 2) contains the sequence ('dashed_seq').
        cns_file (str): Path to the file containing ancestral sequences.

    Returns:
        pd.DataFrame: A DataFrame with duplicate sequences removed, keeping the sequence with the
                      highest identity to the ancestral sequence.

    Raises:
        FileNotFoundError: If the specified `cns_file` does not exist.
        KeyError: If the DataFrame does not contain the expected columns.
        ValueError: If `get_ancestral_seq` or `compute_identity`/`realign_and_identity` raise value errors.
        TypeError: If `compute_identity`/`realign_and_identity` raise type errors.

    Example:
        Assuming df is a DataFrame with sequence data and CNS_FILENAME exists:
        >>> cleaned_df = drop_duplicates_by_identity(df)
    """   
    # Check if there are duplicates in the DataFrame
    if not df.duplicated(subset=2).any():
        return df  # If no duplicates, return the original DataFrame
    
    df = df.drop_duplicates(subset=[df.columns[1], df.columns[11]]) # remove duplicates of association (single
                                                                    # CNS locus associated with multiple genes)
    
    # Get CNS id from map df
    cns_id = df.iloc[0, 0]
    
    # Get ancestral seq from cns file
    try:
        anc_seq = get_ancestral_seq(cns_id, cns_file)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"CNS file not found: {cns_file}") from e
    except ValueError as e:
        raise ValueError(f"Error getting ancestral sequence: {e}") from e
    
    # Extract duplicates and singles
    dupes = df[df.duplicated(subset=2, keep=False)].copy()
    singles = df.drop_duplicates(subset=2, keep=False)
    
      # Calculate identity to ancestral sequence
    try:
        dupes['Identity'] = dupes.apply(lambda row: compute_identity(anc_seq, row['dashed_seq']), axis=1)
    except:
        try:
            dupes['Identity'] = dupes.apply(lambda row: realign_and_identity(anc_seq, row['dashed_seq']), axis=1)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Error computing identity or realigning: {e}")

    # Sort by identity and species, then remove duplicates keeping the first
    dupes = dupes.sort_values(by=[2, 'Identity'], ascending=[True, False])
    dupes = dupes.drop_duplicates(subset=2, keep='first')
    
    # Unite singles with filtered duplicates
    result = pd.concat([singles, dupes.iloc[:, :13]])
    return result

def grep_cns_file(cns_id, input_file, output_folder=OUTPUT_FOLDER):
    """
    Extracts rows containing the specified CNS ID from a map file using grep.
    
    Args:
        cns_id (str): The CNS ID to search for.
        input_file (str): Path to the map file.
        output_folder (str): Path to the output folder.
    """
    output_file = os.path.join(output_folder, f'{cns_id}.map.csv')
    with open(output_file, "w") as out:
        subprocess.run(["grep", "-w", cns_id, input_file], stdout=out, check=True)


def get_ref_length(cns_id, cns_file):
    """
    Returns the length of the ancestral CNS (Conserved Non-coding Sequence) sequence.
    
    This function retrieves the ancestral sequence corresponding to the given CNS ID 
    and calculates its length. The length serves as a reference for reconstruction 
    purposes.

    Parameters:
    cns_id (str): The identifier of the CNS whose ancestral sequence length is to be determined.

    Returns:
    int: The length of the ancestral CNS sequence.
    """
    return len(get_ancestral_seq(cns_id, cns_file))
 
def importCNSmap(cns_id, cns_file, map_file, output_folder=OUTPUT_FOLDER):
    """
    Imports all occurrences of a given Conserved Non-coding Sequence (CNS),
    formats them for sequence reconstruction, and removes duplicate CNS occurrences 
    within the same genome based on identity to the ancestral sequence.

    The function:
    1. Extracts CNS occurrences using `grep_cns_file`.
    2. Reads the extracted CNS file into a DataFrame.
    3. Formats the sequences by padding them for alignment.
    4. Drops duplicate CNS occurrences from the same genome based on identity.
    5. Returns the processed DataFrame and the file name.

    Parameters:
    cns_id (str): The identifier of the CNS to be imported and processed.
    output_folder (str, optional): The directory where the CNS file is stored. 
                                   Defaults to OUTPUT_FOLDER.

    Returns:
    tuple:
        - pd.DataFrame: A DataFrame containing the processed CNS occurrences.
        - str: The path to the saved CNS file.
    """
    
    # Extract occurrences of the CNS
    grep_cns_file(cns_id, map_file ,output_folder=output_folder)
    
    # Define the file path
    file_name = os.path.join(output_folder,f"{cns_id}.map.csv")
    
    # Read the extracted file into a DataFrame
    df = pd.read_csv(file_name, header=None)
    df.columns = [x for x in range(1, 13)]  # Assign column names
    
    # Format sequences by adding leading dashes based on column 6 and concatenating column  (seuqence)
    df['dashed_seq'] = df.apply(lambda row: '-' * int(row[6]) + row[7], axis=1)
    
    # Ensure all sequences are of equal length by padding with dashes
    total_len = max(df.dashed_seq.str.len())
    df['dashed_seq'] = df['dashed_seq'].apply(lambda x: x + (total_len - len(x)) * '-')
    
    # Remove duplicate CNS occurrences based on identity
    df = drop_duplicates_by_identity(df, cns_file)
    
    return df, file_name


def change_node_name(tree, old_name, new_name):
    """
    Change the name of a node in a BioPython Phylo tree.
    
    Parameters:
        tree (Bio.Phylo.BaseTree.Tree): The Phylo tree object.
        old_name (str): The current name of the node to change.
        new_name (str): The new name for the node.
    
    Returns:
        None
    """
    # Find the node you want to change the name of
    target_node = tree.find_any(old_name)
    
    # If the node is not found, print a message and return
    if target_node is None:
        return
    
    # Change the name of the node
    target_node.name = new_name
    
def trim_tree(tree, species):
    """
    Trims a phylogenetic tree to retain only the specified species.

    This function modifies a given phylogenetic tree by removing all leaves 
    that are not present in the provided list of species. Before pruning, 
    it also corrects a specific node name ('Graimondii_fixed' to 'Graimondii.fixed').

    Parameters:
    tree (Bio.Phylo.BaseTree.Tree): The original phylogenetic tree to be processed.
    species (list of str): A list of species names that should be retained in the tree.

    Returns:
    Bio.Phylo.BaseTree.Tree: A trimmed version of the input tree containing only the 
                             specified species.
    """
    # Ensure the tree contains the correct node naming convention
    change_node_name(tree, 'Graimondii_fixed', 'Graimondii.fixed')
    
    # Create a copy of the tree (optional, depending on whether you need to preserve the original)
    trimmed_tree = tree

    # Iterate through leaves and prune those not in the species list
    for leaf in trimmed_tree.get_terminals():
        if leaf.name not in species:
            trimmed_tree.prune(leaf)

    return trimmed_tree


def write_fasta(names, sequences, fasta_file):
    """
    Writes sequences to a FASTA file.

    This function takes a list of sequence names (identifiers) and their corresponding 
    sequences, and writes them in FASTA format to the specified file.

    Parameters:
    names (list of str): A list of sequence identifiers (headers for the FASTA file).
    sequences (list of str): A list of nucleotide or protein sequences corresponding to each name.
    fasta_file (str): The output file path where the FASTA file will be saved.

    Returns:
    None
    """
    with open(fasta_file, 'w') as f:
        for name, sequence in zip(names, sequences):
            f.write(f'>{name}\n{sequence}\n')

def df_to_fasta(df, cns_id, output_folder=OUTPUT_FOLDER):
    """
    Creates a FASTA file containing all CNS occurrences after filtering 
    and formatting for sequence reconstruction.

    This function extracts sequence names and corresponding processed 
    sequences from a DataFrame and writes them in FASTA format to an output file. 
    The sequences have already been filtered and formatted for reconstruction
    purposes.

    Parameters:
    df (pd.DataFrame): A DataFrame containing sequence names (column 2) and 
                       formatted sequences ('dashed_seq' column).
    cns_id (str): The identifier of the CNS, used to name the output FASTA file.
    output_folder (str, optional): The directory where the FASTA file will be saved. 
                                   Defaults to OUTPUT_FOLDER.

    Returns:
    None
    """
    names = df[2]  # Extract sequence names
    sequences = df['dashed_seq']  # Extract formatted sequences
    fasta_path = os.path.join(output_folder,f"{cns_id}.fasta")
    write_fasta(names, sequences, fasta_path)  # Write sequences to a FASTA file

def fasta_to_dict(file_path):
    """
    Imports a FASTA file into a dictionary where keys are sequence identifiers 
    and values are the corresponding sequences.

    This function reads a FASTA-formatted file and stores the sequences in a dictionary.
    Each sequence identifier (header without '>') is a key, and the corresponding 
    sequence is stored as the value.

    Parameters:
    file_path (str): The path to the FASTA file to be imported.

    Returns:
    dict: A dictionary where:
          - Keys (str) are sequence identifiers from the FASTA file.
          - Values (str) are the corresponding sequences.
    """
    sequences = {}
    current_sequence_name = None
    current_sequence = ""

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                # Store previous sequence before moving to a new identifier
                if current_sequence_name:
                    sequences[current_sequence_name] = current_sequence
                # Extract sequence name (excluding '>')
                current_sequence_name = line[1:]
                current_sequence = ""
            else:
                # Append sequence lines (handles multi-line FASTA sequences)
                current_sequence += line

        # Add the last sequence
        if current_sequence_name:
            sequences[current_sequence_name] = current_sequence

    return sequences

def reconstructed_fasta_to_dict(cns_id, path=DATA_FOLDER + 'sequences'):
    """
    Imports a reconstructed CNS FASTA file into a dictionary.

    This function reads a FASTA file containing reconstructed CNS sequences 
    and stores them in a dictionary, where the sequence identifiers serve as 
    keys and the corresponding sequences as values.

    Parameters:
    cns_id (str): The identifier of the CNS for which the reconstructed 
                  sequences should be loaded.
    path (str, optional): The directory where the reconstructed FASTA file 
                          is stored. Defaults to DATA_FOLDER + 'sequences/'.

    Returns:
    dict: A dictionary where:
          - Keys (str) are sequence identifiers from the FASTA file.
          - Values (str) are the corresponding sequences.
    """
    file_path = os.path.join(path, f"{cns_id}.reconstructed.fasta")
    return fasta_to_dict(file_path)

def dict_to_reconstructed_fasta(seq_dict, cns_id, path=OUTPUT_FOLDER):
    """
    Writes a dictionary of reconstructed CNS sequences to a FASTA file.

    This function takes a dictionary where sequence identifiers are keys 
    and sequences are values, and writes them into a FASTA-formatted file.

    Parameters:
    seq_dict (dict): A dictionary containing:
                     - Keys (str): Sequence identifiers.
                     - Values (str): Corresponding nucleotide or protein sequences.
    cns_id (str): The identifier of the CNS, used to name the output FASTA file.
    path (str, optional): The directory where the FASTA file will be saved. 
                          Defaults to OUTPUT_FOLDER.

    Returns:
    None
    """
    file_path = os.path.join(path,f"{cns_id}.reconstructed.fasta")

    with open(file_path, 'w') as fasta_file:
        for name, sequence in seq_dict.items():
            fasta_file.write(f'>{name}\n{sequence}\n')

def trim_annotated_tree(tree_file, cns_id, path=OUTPUT_FOLDER):
    """
    Creates a trimmed and annotated copy of a phylogenetic tree for sequence reconstruction.

    This function modifies an annotated phylogenetic tree so that it matches the 
    unannotated version used by the FastML reconstruction tool. Since FastML 
    cannot handle node names, this function trims the annotated tree to include 
    only the species present in the unannotated version.

    Parameters:
    tree_file (str): Path to the original annotated phylogenetic tree in Newick format.
    cns_id (str): The identifier of the CNS, used to name the output tree file.
    path (str, optional): The directory where the output tree files are stored.
                          Defaults to OUTPUT_FOLDER.

    Returns:
    Phylo.BaseTree.Tree: The trimmed and annotated phylogenetic tree.
    
    Output Files:
    - `{cns_id}.trimmed.annotated.tree`: The annotated trimmed tree for FastML.
    """

    # Define the path to the unannotated trimmed tree
    unannotated_trimmed_tree_file = os.path.join(path,f'{cns_id}.trimmed.tree')

    # Read the annotated tree
    tree = Phylo.read(tree_file, 'newick')

    # Adjust node names to match expected format
    change_node_name(tree, 'Graimondii_fixed', 'Graimondii.fixed')

    # Read the unannotated trimmed tree (used as reference)
    unannotated_trimmed_tree = Phylo.read(unannotated_trimmed_tree_file, 'newick')

    # Extract species from the unannotated tree
    species = [leaf.name for leaf in unannotated_trimmed_tree.get_terminals()]

    # Trim the annotated tree to match the species present in the unannotated tree
    annotated_trimmed_tree = trim_tree(tree, species)

    # Save the trimmed annotated tree in Newick format
    output_tree_file = os.path.join(path,f'{cns_id}.trimmed.annotated.tree')
    Phylo.write(annotated_trimmed_tree, output_tree_file, "newick")

    return annotated_trimmed_tree

def extract_node_names(tree):
    """
    Extracts all node names from a phylogenetic tree.

    This function retrieves the names of all clades (both internal and leaf nodes) 
    from a given phylogenetic tree.

    Parameters:
    tree (Bio.Phylo.BaseTree.Tree): A phylogenetic tree in Biopython's Phylo format.

    Returns:
    list of str: A list of node names, including both internal and leaf nodes.
                 Nodes without a name will be returned as None.
    """
    return [clade.name for clade in tree.find_clades()]

def change_dictionary_nodes(sequences, fastml_tree, annotated_tree):
    """
    Maps sequence node names from the FastML output tree to their corresponding names 
    in the annotated phylogenetic tree.

    This function ensures that the reconstructed sequences from FastML, which assigns 
    arbitrary node names, are correctly matched to their respective nodes in the 
    annotated phylogenetic tree.

    Parameters:
    sequences (dict): A dictionary where:
                      - Keys (str) are node names from the FastML tree.
                      - Values (str) are the corresponding reconstructed sequences.
    fastml_tree (Bio.Phylo.BaseTree.Tree): The tree output by FastML, containing 
                                           arbitrary node names.
    annotated_tree (Bio.Phylo.BaseTree.Tree): The corresponding annotated tree 
                                              with meaningful node names.

    Returns:
    dict: A new dictionary where:
          - Keys (str) are the meaningful node names from the annotated tree.
          - Values (str) are the reconstructed sequences from the FastML output.
    """

    # Create a mapping between FastML's arbitrary node names and annotated tree names
    mapping = dict(zip(extract_node_names(fastml_tree), extract_node_names(annotated_tree)))

    # Generate a new dictionary with updated node names
    new_dict = {mapping[key]: sequences[key] for key in sequences.keys()}

    return new_dict

def run_FastML(cns_id, path_to_binary=FASTML_BINARY_PATH, output_folder=OUTPUT_FOLDER):
    """
    Runs the FastML sequence reconstruction program for a given CNS.

    This function constructs and executes a command-line call to FastML, a program 
    for ancestral sequence reconstruction. It takes a CNS identifier and uses 
    the corresponding sequence and tree files to perform the reconstruction.

    Parameters:
    cns_id (str): The identifier of the CNS for which ancestral reconstruction 
                    should be performed.
    path_to_binary (str, optional): The path to the FastML binary. 
                                    Defaults to FASTML_BINARY_PATH.
    output_folder (str, optional): The directory where input and output files are stored. 
                                    Defaults to OUTPUT_FOLDER.

    Outputs:Outputs:
    - `{cns_id}.fastml.tree`: The reconstructed phylogenetic tree with sequences.
    - `{cns_id}.reconstructed.fasta`: The reconstructed ancestral sequences in FASTA format.
    - `{cns_id}.fastml.log`: Log file containing output and errors from FastML.
    
    Returns:
    None
    """

    command = [
        path_to_binary,
        "-qf",  # Quiet mode (fewer messages)
        "-mh",  # Maximum likelihood model
        "-s", os.path.join(output_folder, f"{cns_id}.fasta"),  # Input sequence file
        "-t", os.path.join(output_folder, f"{cns_id}.trimmed.tree"),  # Input tree file
        "-y", "/dev/null",  # Suppresses a specific output
        "-d", "/dev/null",  # Suppresses a specific output
        "-x", os.path.join(output_folder, f"{cns_id}.fastml.tree"),  # Output tree file
        "-k", "/dev/null",  # Suppresses a specific output
        "-e", "/dev/null",  # Suppresses a specific output
        "-j", os.path.join(output_folder, f"{cns_id}.reconstructed.fasta")  # Output reconstructed FASTA file
    ]

    log_file_path = os.path.join(output_folder, f"{cns_id}.fastml.log")


    print(f"Reconstructing ancestral sequences for {cns_id}")
    try:
        # Execute the command
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        # Write output to the log file
        with open(log_file_path, "w") as log_file:
            log_file.write("Output:\n")
            log_file.write(result.stdout)
            if result.stderr:
                log_file.write("\nError:\n")
                log_file.write(result.stderr)
    
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running FastML: {e}")
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def fix_reconstructed_sequences_annotation(cns_id, sequences, annotated_trimmed_tree, output_folder):
    """
    Corrects the annotation of reconstructed sequences by mapping FastML's 
    arbitrary node names to the corresponding names in the annotated phylogenetic tree.

    This function reads the FastML output tree, which contains arbitrary node names, 
    and remaps the sequence dictionary to use the meaningful node names from the 
    annotated trimmed tree.

    Parameters:
    cns_id (str): The identifier of the CNS for which sequences were reconstructed.
    sequences (dict): A dictionary containing:
                      - Keys (str): Node names from the FastML tree.
                      - Values (str): Corresponding reconstructed sequences.
    annotated_trimmed_tree (Bio.Phylo.BaseTree.Tree): The reference phylogenetic 
                                                      tree with meaningful node names.
    output_folder (str): The directory where FastML output files are stored.

    Returns:
    dict: A dictionary where:
          - Keys (str) are corrected node names from the annotated trimmed tree.
          - Values (str) are the reconstructed sequences from the FastML output.
    """

    # Define the path to the FastML output tree
    fastml_tree_file = os.path.join(output_folder,f'{cns_id}.fastml.tree')

    # Read the FastML output tree (which contains arbitrary node names)
    fastml_tree = Phylo.read(fastml_tree_file, format='newick')

    # Correct the node names in the sequence dictionary
    sequences = change_dictionary_nodes(sequences, fastml_tree, annotated_trimmed_tree)

    return sequences

def import_reconstructed_sequences(cns_id, annotated_trimmed_tree, output_folder=OUTPUT_FOLDER):
    """
    Imports and corrects the annotation of reconstructed CNS sequences.

    This function:
    1. Loads the reconstructed sequences from a FASTA file.
    2. Corrects node names in the sequence dictionary to match the annotated trimmed tree.
    3. Saves the corrected sequences back to a FASTA file.

    Parameters:
    cns_id (str): The identifier of the CNS for which sequences were reconstructed.
    annotated_trimmed_tree (Bio.Phylo.BaseTree.Tree): The reference phylogenetic 
                                                      tree with meaningful node names.
    output_folder (str, optional): The directory where input and output files are stored. 
                                   Defaults to OUTPUT_FOLDER.

    Returns:
    dict: A dictionary where:
          - Keys (str) are the corrected node names from the annotated trimmed tree.
          - Values (str) are the reconstructed sequences.
    
    Output:
    - `{cns_id}.reconstructed.fasta`: A corrected FASTA file with the reconstructed sequences.
    """

    # Load reconstructed sequences from FASTA file
    sequences = reconstructed_fasta_to_dict(cns_id, output_folder)

    # Fix sequence annotation (map FastML node names to meaningful names)
    sequences = fix_reconstructed_sequences_annotation(cns_id, sequences, annotated_trimmed_tree, output_folder)

    # Save the corrected sequences back to a FASTA file
    dict_to_reconstructed_fasta(sequences, cns_id, output_folder)

    return sequences


def validate_data_folders(data_folder):
    """
    Ensures that the required data directories exist, creating them if necessary.

    This function checks for the existence of the 'sequences/' and 'trees/' 
    subdirectories within the specified data folder. If they do not exist, 
    they are created.

    Parameters:
    data_folder (str): The root directory where the sequences and trees folders should be located.

    Returns:
    tuple:
        - sequences_folder (str): The path to the 'sequences/' directory.
        - tree_folder (str): The path to the 'trees/' directory.
    """

    # Define paths for sequences and trees subdirectories
    sequences_folder = os.path.join(data_folder, 'sequences/')
    tree_folder = os.path.join(data_folder, 'trees/')
    fastml_log_folder = os.path.join(data_folder, 'fastml_log/')

    # Create the directories if they do not exist
    os.makedirs(sequences_folder, exist_ok=True)
    os.makedirs(tree_folder, exist_ok=True)
    os.makedirs(fastml_log_folder, exist_ok=True)


    return sequences_folder, tree_folder, fastml_log_folder
    

def move_reconstruction_files_to_folder(cns_id, output_folder):
    """
    Moves reconstructed sequence and tree files to the appropriate data folders.

    This function ensures that reconstructed sequences and annotated tree files 
    are placed in their respective folders within the specified data directory.

    Parameters:
    cns_id (str): The identifier of the CNS for which files are being moved.
    output_folder (str): The directory where the output files are initially stored.

    Returns:
    None
    """

    # Validate and retrieve paths for sequence and tree folders
    sequence_folder, tree_folder, fastml_log_folder = validate_data_folders(output_folder)

    # Define file paths
    reconstructed_fasta_file = os.path.join(output_folder,f"{cns_id}.reconstructed.fasta")
    annotated_tree_file = os.path.join(output_folder,f"{cns_id}.trimmed.annotated.tree")
    fastml_log = os.path.join(output_folder,f"{cns_id}.fastml.log")

    # Move files to their respective folders
    for file, folder in [(reconstructed_fasta_file, sequence_folder), (annotated_tree_file, tree_folder),
                         (fastml_log, fastml_log_folder)]:
        if os.path.exists(file):  # Ensure the file exists before moving
            os.rename(file, file.replace(output_folder, folder))


def sequence_reconstruction_cleanup(cns_id, output_folder=OUTPUT_FOLDER, data_folder=None):
    """
    Cleans up intermediate files generated during sequence reconstruction.

    This function:
    1. Moves final reconstructed sequences and trees to the specified data folder (if provided).
    2. Deletes unnecessary files that were used during FastML sequence reconstruction.

    Parameters:
    cns_id (str): The identifier of the CNS for which reconstruction was performed.
    output_folder (str, optional): The directory where intermediate and final files are stored. 
                                   Defaults to OUTPUT_FOLDER.
    data_folder (str, optional): The root directory where final files should be moved. 
                                 If None, files are not moved.

    Returns:
    None
    """

    move_reconstruction_files_to_folder(cns_id, output_folder)
    # Define paths for temporary/intermediate files
    fastml_tree_file = os.path.join(output_folder,f"{cns_id}.fastml.tree")
    unannotated_trimmed_tree_file = os.path.join(output_folder,f"{cns_id}.trimmed.tree")
    map_fasta_file = os.path.join(output_folder,f"{cns_id}.fasta")

    # Remove intermediate files if they exist
    for file in [fastml_tree_file, unannotated_trimmed_tree_file, map_fasta_file]:
        if os.path.exists(file):  # Ensure the file exists before deleting
            os.remove(file)

def remove_dashes_from_sequences(sequences):
    """
    Removes dashes from all aligned sequences to prepare them for the energy model.

    This function is used to process aligned sequences that contain gap characters ('-'),
    ensuring that the sequences are suitable for energy model analysis, which requires 
    ungapped sequences.

    Parameters:
    sequences (dict): A dictionary where:
                      - Keys (str): Sequence identifiers.
                      - Values (str): Aligned sequences containing dashes.

    Returns:
    dict: A dictionary where:
          - Keys (str): Sequence identifiers.
          - Values (str): Sequences with dashes removed.
    """

    # Remove dashes from all sequences to make them compatible with the energy model
    return {key: seq.replace('-', '') for key, seq in sequences.items()}
