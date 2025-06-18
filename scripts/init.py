import argparse
import os
import pandas as pd
from Bio import Phylo

from config import OUTPUT_FOLDER, DATA_FOLDER, FASTML_BINARY_PATH, CNS_FILE, MAP_FILE, UNANNOTATED_TREE, ANNOTATED_TREE
from reconstruct import *
from create_prob_mat import compute_tree_mutation_matrix


def initialize_data(cns_id, cns_file, map_file, tree_file, output_folder=OUTPUT_FOLDER):
    """
    Initializes data for sequence reconstruction.

    This function performs the following steps:
    1.  Creates the output folder if it doesn't exist.
    2.  Imports CNS (Conserved Non-coding Sequence) data and mapping information using `importCNSmap`.
    3.  Filters out 'Hcannabinus' species from the CNS data, as it's not present in the tree file.
    4.  Extracts unique species names from the CNS data.
    5.  Reads the phylogenetic tree from the given tree file.
    6.  Trims the tree to include only the species present in the CNS data using `trim_tree`.
    7.  Exports the trimmed tree to a newick format file.
    8.  Converts the CNS data to a FASTA format file using `df_to_fasta`.
    9.  Removes the temporary CNS map file.

    Args:
        cns_id (str): The ID of the CNS.
        cns_file (str): Path to the CNS file.
        map_file (str): Path to the MAP file.
        tree_file (str): Path to the phylogenetic tree file.
        output_folder (str, optional): Path to the output folder. Defaults to `OUTPUT_FOLDER`.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    df_cns, cns_map_file = importCNSmap(cns_id, cns_file, map_file, output_folder)
    df_cns = df_cns[df_cns.loc[:, 2] != 'Hcannabinus']  # Hcannabinus not found in tree file, causes fastml to crash
    species = df_cns.loc[:, 2].unique()
    tree = Phylo.read(tree_file, "newick")
    trimmed_tree = trim_tree(tree, species)

    # Export trimmed tree
    trimmed_tree_path = os.path.join(output_folder, f"{cns_id}.trimmed.tree")
    Phylo.write(trimmed_tree, trimmed_tree_path, "newick")

    # Export fasta file with sequences after completing dashes
    df_to_fasta(df_cns, cns_id, output_folder)
    os.remove(cns_map_file)


def reconstruct_sequences(cns_id, annotated_tree_file, path_to_binary=FASTML_BINARY_PATH, output_folder=OUTPUT_FOLDER,
                         data_folder=None):
    """
    Reconstructs ancestral sequences using FastML.

    This function performs the following steps:
    1.  Trims the annotated tree to match the CNS data using `trim_annotated_tree`.
    2.  Runs FastML to reconstruct ancestral sequences using `run_FastML`.
    3.  Imports the reconstructed sequences and associates them with the tree nodes using `import_reconstructed_sequences`.
    4.  Cleans up temporary files generated during the reconstruction process using `sequence_reconstruction_cleanup`.
    5.  Optionally removes dashes from the reconstructed sequences.

    Args:
        cns_id (str): The ID of the CNS.
        annotated_tree_file (str): Path to the annotated phylogenetic tree file.
        path_to_binary (str, optional): Path to the FastML binary. Defaults to `FASTML_BINARY_PATH`.
        output_folder (str, optional): Path to the output folder. Defaults to `OUTPUT_FOLDER`.
        data_folder (str, optional): Path to the data folder. Defaults to `None`.

    Returns:
        tuple: A tuple containing the trimmed annotated tree and a dictionary of reconstructed sequences.
    """
    annotated_trimmed_tree = trim_annotated_tree(annotated_tree_file, cns_id, output_folder)
    run_FastML(cns_id, path_to_binary, output_folder)
    sequences = import_reconstructed_sequences(cns_id, annotated_trimmed_tree, output_folder)
    sequence_reconstruction_cleanup(cns_id, output_folder, data_folder)
    # sequences = remove_dashes_from_sequences(sequences)
    return annotated_trimmed_tree, sequences

def create_mutation_matrix(cns_id, tree, sequences, output_folder=OUTPUT_FOLDER):
    """
    Generates a mutation  matrix for a given CNS
    based on the provided phylogenetic tree and reconstructed sequences.
    The result is a matrix with a count of observed substitutions from each basepair.

    The resulting matrix is saved as a CSV file within a dedicated 'mutation_matrices' 
    subfolder within the specified output folder to create a mutation probability matrix using 
    'create_prob_mat.py'.

    Args:
        cns_id (str): A unique identifier for the CNS.
        tree (Bio.Phylo.BaseTree.Tree): The relevant trimmed tree of a CNS.
        sequences (dict): A data structure containing the reconstructed
            sequences for each node in the tree.
        output_folder (str, optional): The directory where the mutation matrices will be saved.
            Defaults to the value of the global variable 'OUTPUT_FOLDER'.

    Returns:
        None. The function saves the mutation matrix to a CSV file.

    Raises:
        FileNotFoundError: If the provided 'output_folder' does not exist and cannot be created.
        TypeError: If the input 'tree' or 'sequences' are not in the expected format for
            'compute_tree_mutation_matrix'.
        Exception: If any other error occurs during the matrix computation or file saving.

    Note:
        - The mutation matrix is saved as a CSV file with the naming convention
          "{cns_id}.mut.mat.csv".
        - The 'mutation_matrices' subfolder is created within the specified output folder
          if it does not already exist.
    """
    matrices_folder = os.path.join(output_folder, 'mutation_matrices/')
    if not os.path.exists(matrices_folder):
        os.makedirs(matrices_folder)

    mutation_matrix_file = os.path.join(matrices_folder, f"{cns_id}.mut.mat.csv")

    mutation_matrix = compute_tree_mutation_matrix(tree, sequences)  # Assumes this function is defined elsewhere.
    mutation_matrix.to_csv(mutation_matrix_file)


def main(cns_id, cns_file, map_file, unannotated_tree, annotated_tree, output_folder, data_folder, fastml_binary_path, calc_mut_mat):
    """
    Main function to execute sequence reconstruction and mutation matrix computation.

    This function orchestrates the sequence reconstruction process, calling `initialize_data` and
    `reconstruct_sequences`. It also optionally computes and saves the mutation probability matrix.

    Args:
        cns_id (str): The ID of the CNS.
        cns_file (str): Path to the CNS file.
        map_file (str): Path to the MAP file.
        unannotated_tree (str): Path to the unannotated phylogenetic tree file.
        annotated_tree (str): Path to the annotated phylogenetic tree file.
        output_folder (str): Path to the output folder.
        data_folder (str): Path to the data folder.
        fastml_binary_path (str): Path to the FastML binary.
        calc_mut_mat (bool): If True, calculate and save the mutation probability matrix.
    """
    initialize_data(cns_id, cns_file, map_file, unannotated_tree, output_folder)
    tree, sequences = reconstruct_sequences(cns_id, annotated_tree, fastml_binary_path, output_folder, data_folder)
    if calc_mut_mat:
        create_mutation_matrix(cns_id, tree, sequences, output_folder)


if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Run sequence reconstruction and mutation matrix computation for a single CNS.")

    # Define required argument
    parser.add_argument(
        "--cns_id",
        type=str,
        required=True,
        help="CNS ID to process (required)"
    )

    # Define optional flagged arguments with defaults
    parser.add_argument(
        "--fastml_binary_path",
        type=str,
        default=FASTML_BINARY_PATH,
        help=f"Path to the FastML binary (default: '{FASTML_BINARY_PATH}')"
    )
    parser.add_argument(
        "--cns_file",
        type=str,
        default=CNS_FILE,
        help=f"Path to the CNS file (default: '{CNS_FILE}')"
    )
    parser.add_argument(
        "--map_file",
        type=str,
        default=MAP_FILE,
        help=f"Path to the MAP file (default: '{MAP_FILE}')"
    )
    parser.add_argument(
        "--calc_mut_mat",
        action='store_true',
        default=False,
        help="Calculate and save the mutation count matrix.  "
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(
        cns_id=args.cns_id,
        cns_file=args.cns_file,
        map_file=args.map_file,
        unannotated_tree=UNANNOTATED_TREE,
        annotated_tree=ANNOTATED_TREE,
        output_folder=OUTPUT_FOLDER,
        data_folder=DATA_FOLDER,
        fastml_binary_path=args.fastml_binary_path,
        calc_mut_mat=args.calc_mut_mat
    )