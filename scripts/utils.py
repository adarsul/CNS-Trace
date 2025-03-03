import csv
import pandas as pd
from Bio import Phylo
from Bio import Align
import subprocess
import os
import pickle

OUTPUT_FOLDER = '../output/'
DATA_FOLDER = '../data/'
FASTML_BINARY_PATH = '/workspace/conservatory/scripts/'
CNS_FILENAME = 'Angiosperms.V10.1.cns.csv'
MAP_FILENAME = 'Angiosperms.V10.2.singlepos.map.csv'

def get_cns_ids(file_path):
    with open(file_path, 'r') as file:
        strings_list = file.readlines()  # Each line as an element in the list

    # Remove newline characters 
    strings_list = [line.strip() for line in strings_list]
    return strings_list

def import_tree_and_sequences(cns_id, data_folder = DATA_FOLDER):
    sequences = reconstructed_fasta_to_dict(cns_id, data_folder + 'sequences/')
    tree = Phylo.read(f'{data_folder}trees/{cns_id}.trimmed.annotated.tree', format = 'newick')
    return tree, sequences

def get_ancestral_seq(cns_id, cns_file = f'../{CNS_FILENAME}'):
    with open(cns_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if cns_id in row:
                return row[7]  # Assuming 8th column index is 7 (0-indexed)
    return None  # If string is not found in any row

def compute_identity(seq1, seq2):
    """
    Calculate the identity percentage between two sequences, ignoring gaps ('-').
    
    Args:
        seq1 (str): The first sequence.
        seq2 (str): The second sequence.

    Returns:
        float: The identity percentage as a value between 0 and 100.
    """
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of the same length.")
    
    match_count = 0
    valid_positions = 0

    for base1, base2 in zip(seq1, seq2):
        if base1 == '-' and base2 == '-':  # Ignore positions with gaps
            continue
        if base1 == base2:
            match_count += 1
            valid_positions +=1
        elif base1 == '-' or base2 == '-':
            match_count -= 0.1
        else:
            valid_positions +=1

    if valid_positions == 0:
        return 0.0  # Avoid division by zero if there are no valid positions

    identity_percentage = (match_count / valid_positions) * 100
    return identity_percentage

def realign_and_identity(ref_seq, seq):
    # Initialize the aligner
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    alignment = aligner.align(ref_seq, seq)
    aligned_a, aligned_b = alignment[0][0], alignment[0][1]
    return compute_identity(aligned_a, aligned_b)


def drop_duplicates_by_identity(df, cns_file=f'../{CNS_FILENAME}'):
    # Check if there are duplicates in the DataFrame
    if not df.duplicated(subset=2).any():
        return df  # If no duplicates, return the original DataFrame
    
    # Get CNS id from map df
    cns_id = df.iloc[0, 0]
    
    # Get ancestral seq from cns file
    anc_seq = get_ancestral_seq(cns_id, cns_file)
    
    # Extract duplicates and singles
    dupes = df[df.duplicated(subset=2, keep=False)].copy()
    singles = df.drop_duplicates(subset=2, keep=False)
    
    # Calculate identity to ancestral sequence 
    try:
        dupes['Identity'] = dupes.apply(lambda row: compute_identity(anc_seq, row['dashed_seq']), axis=1)
    except:
        dupes['Identity'] = dupes.apply(lambda row: realign_and_identity(anc_seq, row['dashed_seq']), axis=1)

    # Sort by identity and species, then remove duplicates keeping the first
    dupes = dupes.sort_values(by=[2, 'Identity'], ascending=[True, False])
    dupes = dupes.drop_duplicates(subset=2, keep='first')
    
    # Unite singles with filtered duplicates
    result = pd.concat([singles, dupes.iloc[:, :13]])
    return result


def grep_cns_file(cns_id, input_file = f"../{MAP_FILENAME}", output_folder = OUTPUT_FOLDER):
    output_file = output_folder + cns_id +'.map.csv'
    # Run the grep command
    with open(output_file, "w") as out:
        subprocess.run(
            ["grep", "-w", cns_id, input_file],
            stdout=out,
            check=True
        )

def get_ref_length(cns_id):
    return len(get_ancestral_seq(cns_id))

def importCNSmap(cns_id, output_folder = OUTPUT_FOLDER):
    grep_cns_file(cns_id, output_folder = output_folder)
    file_name = output_folder + cns_id+'.map.csv'
    df = pd.read_csv(file_name, header = None)
    df.columns = [x for x in range(1,13)]
    df['dashed_seq'] = df.apply(lambda row: '-'*int(row[6]) + row[7], axis = 1)
    total_len = max(df.dashed_seq.str.len())
    df['dashed_seq'] = df['dashed_seq'].apply(lambda x: x + (total_len - len(x)) * '-')  
    df = drop_duplicates_by_identity(df)   
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
    Trims a phylogenetic tree to contain only specified leaves.

    Args:
        tree - Original tree to be processed
        species - A list containing strings of wanted species.
    
    Returns:
        trimmed_tree - A tree that contains only the leaves specified
    """
    change_node_name(tree, 'Graimondii_fixed' ,'Graimondii.fixed')
    trimmed_tree = tree
    for leaf in trimmed_tree.get_terminals():
        if leaf.name not in species:
            trimmed_tree.prune(leaf)
    return trimmed_tree

def write_fasta(names, sequences, fasta_file):
    with open(fasta_file, 'w') as f:
        for name, sequence in zip(names, sequences):
            f.write(f'>{name}\n{sequence}\n')

def df_to_fasta(df, cns_id, output_folder = OUTPUT_FOLDER):
    names = df[2]
    sequences = df['dashed_seq']
    write_fasta(names, sequences, output_folder + cns_id+'.fasta')


def fasta_to_dict(file_path):
    sequences = {}
    current_sequence_name = None
    current_sequence = ""

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if current_sequence_name:
                    sequences[current_sequence_name] = current_sequence
                current_sequence_name = line[1:]
                current_sequence = ""
            else:
                current_sequence += line

        # Add the last sequence
        if current_sequence_name:
            sequences[current_sequence_name] = current_sequence

    return sequences


def reconstructed_fasta_to_dict(cns_id, path = DATA_FOLDER+'sequences/'):
    file_path = f'{path}{cns_id}.reconstructed.fasta'
    return fasta_to_dict(file_path)
    

def dict_to_reconstructed_fasta(seq_dict, cns_id, path = OUTPUT_FOLDER):
    file_path = f'{path}{cns_id}.reconstructed.fasta'
    with open(file_path, 'w') as fasta_file:
        for name, sequence in seq_dict.items():
            fasta_file.write(f'>{name}\n')
            fasta_file.write(f'{sequence}\n')

def trim_annotated_tree(tree_file, cns_id, path = OUTPUT_FOLDER):
    unannotated_trimmed_tree_file = f'{path}{cns_id}.trimmed.tree'
    tree = Phylo.read(tree_file, 'newick')
    change_node_name(tree, 'Graimondii_fixed' ,'Graimondii.fixed')
    unannotated_trimmed_tree = Phylo.read(unannotated_trimmed_tree_file, 'newick')
    species = [leaf.name for leaf in unannotated_trimmed_tree.get_terminals()]
    annotated_trimmed_tree = trim_tree(tree, species)
    Phylo.write(annotated_trimmed_tree, path +cns_id+".trimmed.annotated.tree", "newick")
    return annotated_trimmed_tree

def extract_node_names(tree):
    return [clade.name for clade in tree.find_clades()]

def change_dictionary_nodes(sequences, fastml_tree, anottated_tree):
    mapping = dict(zip(extract_node_names(fastml_tree), extract_node_names(anottated_tree)))
    new_dict = {}
    for key in sequences.keys():
        new_key = mapping[key]
        new_dict[new_key] = sequences[key]
    return new_dict

def run_FastML(cns_id, path_to_binary = FASTML_BINARY_PATH, output_folder = OUTPUT_FOLDER):
    command = [
    f"{path_to_binary}fastml",
    "-qf",
    "-mh",
    "-s", f"{output_folder}{cns_id}.fasta",
    "-t", f"{output_folder}{cns_id}.trimmed.tree",
    "-y", "/dev/null/a",
    "-d", "/dev/null/a",
    "-x", f"{output_folder}{cns_id}.fastml.tree",
    "-k", "/dev/null/a",
    "-d", "/dev/null/a",
    "-e", "/dev/null/a",
    "-j", f"{output_folder}{cns_id}.reconstructed.fasta"
    ]
    
    try:
        # Execute the command
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        # Print the output and errors (if any)
        print("Output:", result.stdout)
        if result.stderr:
            print("Error:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the command: {e}")


def fix_reconstructed_sequences_annotation(cns_id, sequences, annotated_trimmed_tree, output_folder):
    fastml_tree_file = f'{output_folder}{cns_id}.fastml.tree'
    fastml_tree = Phylo.read(fastml_tree_file, format = 'newick')
    sequences = change_dictionary_nodes(sequences, fastml_tree, annotated_trimmed_tree)
    return sequences

def import_reconstructed_sequences(cns_id, annotated_trimmed_tree, output_folder = OUTPUT_FOLDER):
    sequences = reconstructed_fasta_to_dict(cns_id, output_folder)
    sequences = fix_reconstructed_sequences_annotation(cns_id, sequences, annotated_trimmed_tree, output_folder)
    dict_to_reconstructed_fasta(sequences, cns_id, output_folder)
    return sequences

def validate_data_folders(data_folder):
    sequences_folder = data_folder + 'sequences/'
    tree_folder = data_folder + 'trees/'
    if not os.path.exists(sequences_folder): os.makedirs(sequences_folder)
    if not os.path.exists(tree_folder): os.makedirs(tree_folder)
    return sequences_folder, tree_folder
    
def move_files_to_data_folder(cns_id, output_folder, data_folder):
    sequence_folder, tree_folder = validate_data_folders(data_folder)
    reconstructed_fasta_file = f'{output_folder}{cns_id}.reconstructed.fasta'
    annotated_tree_file = f'{output_folder}{cns_id}.trimmed.annotated.tree'
    for file,folder in [(reconstructed_fasta_file, sequence_folder), (annotated_tree_file, tree_folder)]:
        os.rename(file,file.replace(output_folder,folder))    

def sequence_reconstruction_cleanup(cns_id, output_folder = OUTPUT_FOLDER, data_folder = None):
    if data_folder:
        move_files_to_data_folder(cns_id, output_folder, data_folder)
    fastml_tree_file = f'{output_folder}{cns_id}.fastml.tree'
    unannotated_trimmed_tree_file = f'{output_folder}{cns_id}.trimmed.tree'
    map_fasta_file = f'{output_folder}{cns_id}.fasta'
    for file in [fastml_tree_file, unannotated_trimmed_tree_file, map_fasta_file]:
        os.remove(file)

def remove_dashes_from_sequences(sequences):
    for key, seq in sequences.items():
        sequences[key] = seq.replace('-','')
    return sequences


################################################################################################
# Model Utility Functions
################################################################################################

def find_pwm_files(prefix, data_folder):
    """
    Find all files in the specified directory that start with the given prefix.
    
    Args:
        directory (str): The directory to search for files.
        prefix (str): The prefix string to look for at the beginning of file names.
        
    Returns:
        list: A list containing the file names that start with the specified prefix.
    """
    # Initialize an empty list to store file names
    matching_files = []
    
    # Iterate over all files in the directory
    for filename in os.listdir(data_folder):
        # Check if the file name starts with the specified prefix
        if filename.startswith(prefix):
            # Add the file name to the list
            matching_files.append(filename)
    
    # Return the list of files matching the prefix
    return matching_files

def import_pwm(TF):
    try:
        pwm = parse_meme(f'/home/adar/workspace/cns_evolution/data/dap_pwms/meme_{TF}.txt')
    except:
        pwm = parse_meme(f'/home/adar/workspace/cns_evolution/data/dap_pwms/meme_{TF}.pickle')
    return pwm



def parse_meme(meme_file):
    """
    Reads a meme file and returns a PWM where poisitions that contain 0
    are replaced by 0.5 divided by the number of sequences used to produce the PWM
    """
    if meme_file.endswith('.pickle'):
        with open(meme_file, 'rb') as f:
            pwm = pickle.load(f)
            return pwm
        
    # Open the input file and read its contents into a list of strings.
    with open(meme_file, 'r') as f:
        lines = f.readlines()

    # Parse the alphabet (nucleotides) used in the input sequences and the number of sites used to build the PWM.    
    for line in lines:
        if line.startswith('ALPHABET='):
            keys = line.replace('ALPHABET= ','').strip('\n')
            keys = [n for n in keys]

        # Get the number of sequences used to build the PWM.
        if line.startswith('letter-probability'):
            s = line.find('nsites')+len('nsites= ')
            e = line.find('E=')
            nsites = int(line[s:e].strip(' '))
            start = lines.index(line) + 1

    # Build the PWM dictionary.
    pwm = {x:list() for x in keys}
    for line in lines[start:-1]:
        if line.startswith('URL'):
            continue
        data = line.split(' ')
        data = [float(x.strip(' \t\n')) for x in data if x]
        for i in range(len(data)):
            if data[i] == 0:
                # Replace any position with probability 0 with a pseudocount based on the number of sites.
                data[i] = 0.5/nsites
        for i in range(len(data)):
            # Add the probability for each character at the current position to the PWM.
            pwm[keys[i]].append(data[i])
    return pwm 

def pwm_name_from_file(pwm_file):
    pwm_name = pwm_file.split('_')[-1]
    pwm_name = pwm_name.replace('.txt','')
    pwm_name = pwm_name.replace('.pickle','')
    return pwm_name