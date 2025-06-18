import sys
import os
import pickle
import csv
from config import PWM_FOLDER, GC_FILE, THRESHOLDS_FILE

# neutral model utils
import pandas as pd
from config import DATA_FOLDER

def find_pwm_files(pwm_folder = PWM_FOLDER):
    """Finds all PWM files in a directory with a given prefix.

    This function searches a specified directory for files whose names
    start with a given prefix.  It's used to locate PWM files.

    Args:
        prefix (str): The prefix of the filenames to search for (e.g., "meme").
        data_folder (str): The path to the directory to search.

    Returns:
        list[str]: A list of filenames (without the path) that match the
            prefix.  Returns an empty list if no matching files are found.

    Raises:
        FileNotFoundError: If the specified `data_folder` does not exist.

    Example:
        >>> find_pwm_files("jaspar", "/path/to/pwm_directory")
        ['jaspar_MA0004.txt', 'jaspar_MA0006.txt']
    """

    # Ensure the data folder exists
    if not os.path.isdir(pwm_folder):
        raise FileNotFoundError(f"Directory '{pwm_folder}' does not exist.")

    # Find and return files matching the prefix
    return [filename for filename in os.listdir(pwm_folder)]

def parse_meme(meme_file):
    """
    Parses a MEME format file and constructs a Position Weight Matrix (PWM) for nucleotides.

    This function reads a MEME file and returns a PWM where positions containing a 
    probability of 0 are replaced by a pseudocount of 0.5 divided by the number 
    of sequences (nsites) used to construct the PWM.

    Only nucleotide PWMs (A, C, G, T) are processed; non-nucleotide alphabets are ignored.

    If the input file is a pickle file, it directly loads the stored PWM.

    Parameters:
    meme_file (str): Path to the MEME file (.txt or .pickle).

    Returns:
    dict: A PWM dictionary where:
          - Keys (str): Nucleotides ('A', 'C', 'G', 'T').
          - Values (list of float): Probability values per position in the PWM.
    """

    # Read the MEME file contents
    with open(meme_file, 'r') as f:
        lines = f.readlines()

    # Initialize variables
    nucleotides = {'A', 'C', 'G', 'T'}  # Only consider nucleotides
    nsites = None
    pwm = {nuc: [] for nuc in nucleotides}
    start = None

    # Parse MEME file contents
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Extract the number of sequences used to build the PWM
        if line.startswith('letter-probability'):
            parts = line.split()
            for j, part in enumerate(parts):
                if part.startswith("nsites="):
                    try:
                        nsites = int(parts[j + 1])  # Extract nsites value
                        start = i + 1  # Start parsing PWM from the next line
                    except (IndexError, ValueError):
                        raise ValueError(f"Invalid nsites value in {meme_file}")
                    break

    # Validate that nsites was found
    if nsites is None or start is None:
        raise ValueError(f"Failed to extract nsites from {meme_file}")

    # Process PWM matrix
    for line in lines[start:]:
        line = line.strip()
        if not line or line.startswith('URL'):  # Ignore empty lines and metadata
            continue

        # Parse probabilities
        data = [float(x) for x in line.split() if x]

        # Ensure we are processing only nucleotide values
        if len(data) != 4:
            raise ValueError(f"Unexpected PWM format in {meme_file}: expected 4 columns for A, C, G, T")

        # Replace zero values with pseudocount
        data = [0.5 / nsites if val == 0 else val for val in data]

        # Assign probabilities to nucleotides
        for i, nuc in enumerate(['A', 'C', 'G', 'T']):
            pwm[nuc].append(data[i])

    return pwm

def pwm_from_pickle(pickle_file):    
    # If the file is a pickle, load and return it directly
    with open(pickle_file, 'rb') as f:
        return pickle.load(f)


def import_pwm(pwm_file):
    """
    Imports a Position Weight Matrix (PWM) from a specified file.

    This function supports importing PWMs stored in either MEME/text format or as pickled Python dictionaries.
    It checks the file extension to determine the appropriate parsing method.

    Args:
        pwm_file (str): The path to the PWM file.

    Returns:
        dict or None: A dictionary representing the PWM if the file is successfully loaded and parsed.
                      Returns None if the file is not found, or if an error occurs during parsing.

    Raises:
        Exception: If an unexpected error occurs during file processing.

    Example:
        >>> pwm = import_pwm("path/to/my_pwm.meme")
        >>> print(pwm)
        {'A':[0.2,0.2], 'T':[0.2,0.5], 'C':[0.4,0.1], 'G':[0.2,0.3]}

    Supported file formats:
        - .txt or .meme: MEME format (parsed using the `parse_meme` function).
        - .pickle: Pickled Python dictionary (loaded using the `pwm_from_pickle` function).
    """
    try:
        if os.path.exists(pwm_file):
            if pwm_file.endswith('.txt') or pwm_file.endswith('.meme'):
                return parse_meme(pwm_file)
            elif pwm_file.endswith('.pickle'):
                return pwm_from_pickle(pwm_file)
        else:
            print(f"Error: {pwm_file} not found.")
            return None
    except Exception as e:
        print(f"Error loading PWM from {pwm_file}: {e}")
        return None


def pwm_name_from_file(pwm_file):
    """
    Extracts the PWM name from a file name.

    This function assumes that the PWM file follows a naming convention where 
    the transcription factor (TF) name appears after the last underscore ('_') 
    in the file name, and removes any `.txt` or `.pickle` extensions.
    
    Args:
    pwm_file (str): The name or path of the PWM file.

    Returns:
    str: The extracted PWM name (typically the TF name).
    
    Example:
        >>> pwm_name = pwm_name_from_file("path/to/meme_ARF5.meme")
        >>> print(pwm_name)
        'ARF5'
    """

    # Extract the part of the filename after the last underscore
    pwm_name = pwm_file.split('_')[-1]

    # Remove possible file extensions
    pwm_name = pwm_name.replace('.txt', '').replace('.pickle', '')

    return pwm_name

def createPWMdb(pwm_folder = PWM_FOLDER):
    """
    Creates a PWM database (dictionary) from PWM files within a specified folder.

    This function iterates through all PWM files in the given folder, parses them,
    and constructs a dictionary where keys are transcription factor (TF) names
    and values are the corresponding PWM dictionaries.

    Args:
        pwm_folder (str, optional): The path to the folder containing PWM files.
                                   Defaults to the value of the global variable PWM_FOLDER.

    Returns:
        dict: A dictionary representing the PWM database.
              Keys are TF names (strings), and values are PWM dictionaries.

    Raises:
        SystemExit: If the specified `pwm_folder` does not exist.

    Example:
        >>> pwm_database = createPWMdb("path/to/my/pwm/folder")
        >>> if pwm_database:
        ...     print(f"PWM database created with {len(pwm_database)} entries.")
        ...     # Access PWM data using TF name: pwm_database["TF_name"]
        ... else:
        ...     print("Failed to create PWM database.") 
    """
    
    if not os.path.exists(pwm_folder):
        print(f"Error: Folder '{pwm_folder}' not found.")
        sys.exit(1)  # Exit with a non-zero exit code to indicate an error

    # initialize empty dictionary
    pwm_db = {} 
    
    pwm_filenames = find_pwm_files(pwm_folder)
    
    # iterate through meme files 
    for pwm_file in pwm_filenames: 
        # turn meme file into pwm dictionary
        pwm = import_pwm(os.path.join(pwm_folder, pwm_file))       
        # modify pwm file name into TF name
        pwm_name = pwm_name_from_file(pwm_file)
        # add element to dictionary dict['TF'] = pwm
        pwm_db[pwm_name] = pwm 
    
    return pwm_db


def species_bg_freq(gc):
    """Calculates background nucleotide frequencies based on GC-content.

    This function calculates the expected frequencies of A, C, G, and T
    nucleotides given a specified GC-content, assuming equal frequencies of
    A and T, and equal frequencies of G and C.

    Args:
        gc (float): The GC-content (proportion of G and C nucleotides),
            a value between 0 and 1.

    Returns:
        dict: A dictionary where keys are nucleotide bases ('A', 'C', 'G', 'T')
            and values are their respective frequencies (float).

    Example:
        >>> species_bg_freq(0.4)
        {'A': 0.3, 'C': 0.2, 'G': 0.2, 'T': 0.3}

    Notes:
        - Assumes a simple model with equal A/T and G/C frequencies.
    """
    at = 1 - gc
    return {'A': at/2, 'C': gc/2, 'G': gc/2, 'T': at/2}

def create_gc_dict(csv_file=GC_FILE):
    """Creates a dictionary of background nucleotide frequencies from a CSV file.

    This function reads a CSV file containing species names and their
    corresponding GC-contents, calculates the background nucleotide
    frequencies using the `species_bg_freq` function, and returns a
    dictionary mapping species names to these frequencies.

    Args:
        csv_file (str, optional): The *full path* to the CSV file.
            Defaults to GC_FILE (defined in the configuration file).

    Returns:
        dict: A dictionary where keys are species names (str) and values are
            dictionaries representing nucleotide frequencies (as returned by
            `species_bg_freq`).

    Raises:
        FileNotFoundError: If the specified CSV file does not exist.
        ValueError: If the GC-content in the CSV file cannot be converted to a float.

    Example:
        # Assuming your config file sets GC_FILE to "../data/gc_content.csv"
        # and gc_content.csv contains:
        # Species1,0.45
        # Species2,0.6
        >>> gc_dict = create_gc_dict()
        >>> print(gc_dict["Species1"])
        {'A': 0.275, 'C': 0.225, 'G': 0.225, 'T': 0.275}

    Notes:
        - The CSV file is expected to have at least two columns:
            - Column 1: Species name (string).
            - Column 2: GC-content (float).
        - Any rows with fewer than two columns are skipped.
        - Relies on the `species_bg_freq` function.
    """
    result_dict = {}
    file_path = csv_file #No longer needs to join with data_folder

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if len(row) >= 2:  # Ensure at least two columns
                try:
                    gc = float(row[1])
                    result_dict[row[0]] = species_bg_freq(gc)
                except ValueError:
                    raise ValueError(
                        f"Invalid GC-content value in CSV file: '{row[1]}'. "
                        f"Could not convert to float."
                    )
    return result_dict


def importThresDB(thresholds_file = THRESHOLDS_FILE):
    """
    Imports a threshold database from a CSV file.

    This function reads a CSV file containing threshold values for different transcription factors (TFs)
    and creates a dictionary representing the threshold database.

    Args:
        thresholds_file (str, optional): The name of the CSV file containing thresholds.
                                        Defaults to f'path/data/thresholds.csv'.
    Returns:
        dict: A dictionary representing the threshold database.
              Keys are TF names (strings), and values are dictionaries with 'strong_thres',
              'medium_thres', and 'weak_thres' keys, each holding a float value.

    Raises:
        FileNotFoundError: If the specified thresholds file does not exist.

    Example:
        >>> threshold_database = importThresDB()
        >>> if threshold_database:
        ...     print(f"Threshold database loaded with {len(threshold_database)} entries.")
        ...     # Access threshold values using TF name: threshold_database["TF_name"]["strong_thres"]
        ... else:
        ...     print("Failed to load threshold database.")

    Note:
        The CSV file should have the following format:
        TF_name,strong_thres,medium_thres,weak_thres
        TF1,0.8,0.6,0.4
        TF2,0.9,0.7,0.5
        ...

        If the file does not exist, the user should run the create_thresholds.py script to create the file.
    """

    if not os.path.exists(thresholds_file):
        raise FileNotFoundError(f"Thresholds file '{thresholds_file}' not found. Please run the create_thresholds.py script to generate the file.")

    thres_db = {}
    with open(thresholds_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            thres_db[row[0]] = {
                'strong_thres': float(row[1]),
                'medium_thres': float(row[2]),
                'weak_thres': float(row[3])
            }
    return thres_db

def filter_pwm_db(pwm_db, tf_names):
    """
    Filters the PWM database to include only the specified transcription factors.

    Args:
        pwm_db (dict): The original PWM database.
        tf_names (list): A list of transcription factor names to keep.

    Returns:
        dict: A filtered PWM database containing only the specified TFs.
    """
    if not tf_names:
        return pwm_db  # Return the original database if no TFs are specified.

    filtered_pwm_db = {}
    missing_tfs = []
    for tf in tf_names:
        if tf in pwm_db:
            filtered_pwm_db[tf] = pwm_db[tf]
        else:
            missing_tfs.append(tf)

    if not filtered_pwm_db:
        raise ValueError(f"None of the specified TFs exist in the PWM database: {', '.join(tf_names)}")
    elif missing_tfs:
        print(f"Warning: The following TFs were not found in the PWM database: {', '.join(missing_tfs)}")

    return filtered_pwm_db

# Neutral model utils (maybe seperate file)-------------------------------

def get_prob_mat(data_folder = DATA_FOLDER):
  """
  Reads a probability matrix from a CSV file into a pandas DataFrame.

  This function loads a pre-calculated matrix of probabilities from a specified
  folder. The CSV file is expected to be named 'probability_matrix.csv' and
  should contain an index column.

  Args:
    data_folder (str, optional): The path to the folder containing the CSV
                                 file. Defaults to the global constant
                                 DATA_FOLDER.

  Returns:
    pandas.DataFrame: A DataFrame containing the probability matrix, with the
                    first column of the CSV set as the DataFrame index.
  """
  return pd.read_csv(data_folder + 'probability_matrix.csv', index_col=0)