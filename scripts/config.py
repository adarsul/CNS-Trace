import os

# Get the absolute path of the current script (config file)
SCRIPTS_FOLDER = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (main folder)
MAIN_FOLDER = os.path.dirname(SCRIPTS_FOLDER)

# Define output and data folders relative to the main folder
OUTPUT_FOLDER = os.path.join(MAIN_FOLDER, 'output/')
DATA_FOLDER = os.path.join(MAIN_FOLDER, 'data/')

# Construct paths for other files and folders
PWM_FOLDER = os.path.join(DATA_FOLDER, 'pwms/')
PEAKS_SEQ_FOLDER = os.path.join(DATA_FOLDER,'peaks_sequences/')
GC_FILE = os.path.join(DATA_FOLDER, 'gc_content.csv')
THRESHOLDS_FILE = os.path.join(DATA_FOLDER, 'thresholds.csv')

FASTML_BINARY_PATH = os.path.join(SCRIPTS_FOLDER, 'fastml')

CNS_FILE = os.path.join(MAIN_FOLDER,'conservatoryV10.final.cns.csv')
MAP_FILE = os.path.join(MAIN_FOLDER,'conservatoryV10.final.map.csv')
UNANNOTATED_TREE = os.path.join(DATA_FOLDER,'Conservatory.tree') 
ANNOTATED_TREE = os.path.join(DATA_FOLDER,'Conservatory.Annotated.tree')
PROBABILITY_MATRIX = os.path.join(DATA_FOLDER,'probability_matrix.csv')