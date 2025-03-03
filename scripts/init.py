from neutral import *
from utils import *
from Bio import Phylo
import os
import pandas as pd
import argparse


OUTPUT_FOLDER = '../output/'
DATA_FOLDER = '../data/'
FASTML_BINARY_PATH = '/sci/labs/idanef/idanef/backup/conservatory/scripts/'
UNANNOTATED_TREE = '../Conservatory.tree' 
ANNOTATED_TREE = '../Conservatory.Annotated.tree'
CNS_ID_FILE = '../Angiosperms_cns_id'

def initialize_data(cns_id, tree_file, output_folder = OUTPUT_FOLDER):
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    df_cns, cns_map_file = importCNSmap(cns_id, output_folder)
    df_cns = df_cns[df_cns.loc[:,2] != 'Hcannabinus'] # Hcannabinus not found in tree file, causes fastml to crash
    species = df_cns.loc[:,2].unique()
    tree = Phylo.read(tree_file, "newick")
    trimmed_tree = trim_tree(tree, species)
    
    # Export trimmed tree
    Phylo.write(trimmed_tree, output_folder+cns_id+".trimmed.tree", "newick") 

    # Export fasta file with sequences after completing dashes
    df_to_fasta(df_cns, cns_id, output_folder)
    os.remove(cns_map_file)
    
def reconstruct_sequences(cns_id, annotated_tree_file, path_to_binary = FASTML_BINARY_PATH, output_folder = OUTPUT_FOLDER, data_folder = None):
    annotated_trimmed_tree = trim_annotated_tree(annotated_tree_file, cns_id, output_folder)
    run_FastML(cns_id, path_to_binary, output_folder)
    sequences = import_reconstructed_sequences(cns_id, annotated_trimmed_tree, output_folder)
    sequence_reconstruction_cleanup(cns_id, output_folder, data_folder)
#    sequences = remove_dashes_from_sequences(sequences)
    return annotated_trimmed_tree, sequences 

def main(cns_id, unannotated_tree, annotated_tree, output_folder, data_folder, fastml_binary_path):
        initialize_data(cns_id, unannotated_tree, output_folder)
        tree, sequences = reconstruct_sequences(cns_id, annotated_tree, fastml_binary_path, output_folder, data_folder)
        mutation_matrix = compute_tree_mutation_matrix(tree,sequences)
        mutation_matrix.to_csv(f'{output_folder}{cns_id}.mut.mat.csv')

            

    
if __name__ == "__main__":
  # Set up the argument parser
    parser = argparse.ArgumentParser(description="Run sequence reconstruction and mutation matrix computation.")

    # Define required argument
    parser.add_argument(
        "--cns_id", 
        type=str, 
        required=True, 
        help="CNS ID to process (required)"
    )

    # Define optional flagged arguments with defaults
    parser.add_argument(
        "--unannotated_tree", 
        type=str, 
        default=UNANNOTATED_TREE, 
        help="Path to the unannotated tree file (default: '../Conservatory.tree')"
    )
    parser.add_argument(
        "--annotated_tree", 
        type=str, 
        default=ANNOTATED_TREE, 
        help="Path to the annotated tree file (default: '../Conservatory.Annotated.tree')"
    )
    parser.add_argument(
        "--output_folder", 
        type=str, 
        default=OUTPUT_FOLDER, 
        help="Output folder path (default: '../output/')"
    )
    parser.add_argument(
        "--data_folder", 
        type=str, 
        default=DATA_FOLDER, 
        help="Data folder path (default: '../data/')"
    )
    parser.add_argument(
        "--fastml_binary_path", 
        type=str, 
        default=FASTML_BINARY_PATH, 
        help="Path to the FastML binary (default: '/sci/labs/idanef/idanef/backup/conservatory/scripts/')"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(
        cns_id=args.cns_id,
        unannotated_tree=args.unannotated_tree,
        annotated_tree=args.annotated_tree,
        output_folder=args.output_folder,
        data_folder=args.data_folder,
        fastml_binary_path=args.fastml_binary_path
    )