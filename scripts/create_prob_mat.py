import pandas as pd
import os
import argparse
from config import DATA_FOLDER, OUTPUT_FOLDER

MUT_MAT_FOLDER = os.path.join(OUTPUT_FOLDER, 'mutation_matrices/')

def compute_tree_mutation_matrix(tree, sequences):
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
    ignore_chars = ['R', 'Y', 'S', 'W', 'K', 'M', 'B', 'D', 'H', 'V', 'N', 'X','-']
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
    """
    Normalizes each row of a DataFrame by dividing it by the sum of values in that row.

    This function computes row-wise normalization, ensuring that the sum of each row 
    becomes 1. It is useful for converting count data into relative proportions.

    Parameters:
    df (pd.DataFrame): A Pandas DataFrame with numeric values.

    Returns:
    pd.DataFrame: A new DataFrame where each value is divided by the row sum.
    
    Notes:
    - Rows with a sum of 0 will result in NaN values.
    - This assumes that all columns contain numerical values.
    """
    return df.div(df.sum(axis=1), axis=0)

def create_prob_mat(output_folder, data_folder, prob_mat_filename):
    """
    Summarizes mutation matrices from multiple files into a single probability matrix.

    This function reads multiple mutation matrix files (CSV format) from a specified folder.
    Each file contains a count matrix of nucleotide substitutions observed in CNS tree reconstructions.
    The function sums all matrices into a single mutation matrix, then converts the summed counts 
    into a probability matrix by normalizing each row. The resulting matrix is saved as a CSV file.

    Parameters:
    -----------
    output_folder : str
        Path to the folder containing mutation matrix files. These files must have names ending with 'mut.mat.csv'.
    
    data_folder : str
        Path to the folder where the final probability matrix will be saved.

    Process:
    --------
    1. Identify all files in `output_folder` that end with 'mut.mat.csv'.
    2. Initialize an empty 4x4 mutation matrix with nucleotide bases (A, C, G, T) as both rows and columns.
    3. Read and sum all mutation matrices from the identified files.
    4. Normalize the summed matrix row-wise to obtain probability values.
    5. Save the probability matrix as `probability_matrix.csv` in `data_folder`.
    6. Remove the processed mutation matrix files from `output_folder`.
    7. Print the number of processed files (CNS reconstructions).

    Notes:
    ------
    - Assumes that all input matrices have the same format: a 4x4 table with 'A', 'C', 'G', 'T' as row and column labels.

    Example:
    --------
    ```
    create_prob_mat(output_folder='/path/to/mutation_matrices/', 
                    data_folder='/path/to/save/probability_matrix/')
    ```
    """
    mutation_matrices_files = [f for f in os.listdir(output_folder) if f.endswith('mut.mat.csv')]

    if not mutation_matrices_files:
        print(f"No mutation matrix files found in {output_folder}. Exiting.")
        return

    # Initialize an empty mutation matrix with zeros
    mutation_matrix = pd.DataFrame(0, columns=['A', 'C', 'G', 'T'], index=['A', 'C', 'G', 'T'])

    count = 0  # Counter for processed files

    # Loop through each mutation matrix file and sum them
    for file in mutation_matrices_files:
        file_path = os.path.join(output_folder, file)
        try:
            current_mat = pd.read_csv(file_path, index_col=0)

            # Validate expected format
            if not set(current_mat.index) == {'A', 'C', 'G', 'T'} or not set(current_mat.columns) == {'A', 'C', 'G', 'T'}:
                print(f"Warning: Skipping {file} due to unexpected format.")
                continue

            mutation_matrix += current_mat
            count += 1
        except Exception as e:
            print(f"Error processing {file}: {e}")

    # Normalize the summed mutation matrix to obtain probabilities
    if (mutation_matrix.sum(axis=1) == 0).any():
        print("Warning: Some rows have zero total substitutions, which may cause division errors.")

    prob_mat = divide_by_rows(mutation_matrix)  # Ensure divide_by_rows is defined

    # Save the probability matrix to a CSV file
    prob_mat.to_csv(os.path.join(data_folder, prob_mat_filename))

    print(f'Processed {count} CNSs successfully')

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Create a probability matrix from mutation matrix files.")
    parser.add_argument(
        "--output_folder",
        type=str,
        default=MUT_MAT_FOLDER,
        help=f"Output folder path where mutation matrix files are stored (default: {OUTPUT_FOLDER})"
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default=DATA_FOLDER,
        help=f"Data folder path where the probability matrix will be saved (default: {DATA_FOLDER})"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default='probability_matrix.csv',
        help=f"Name of output file (default: probability_matrix.csv)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Validate folder existence
    if not os.path.exists(args.output_folder):
        print(f"Error: Output folder '{args.output_folder}' does not exist.")
        exit(1)
    
    if not os.path.exists(args.data_folder):
        print(f"Error: Data folder '{args.data_folder}' does not exist.")
        exit(1)

    # Call the function with parsed arguments
    create_prob_mat(output_folder=args.output_folder, data_folder=args.data_folder, 
                    prob_mat_filename=args.output_file)
