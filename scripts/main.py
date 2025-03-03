import traceback
import argparse
from Bio import Phylo
from utils import *
from model import *
from neutral import *

DATA_FOLDER = '../data/'
OUTPUT_FOLDER = '../output/'
N_SIMULATIONS = 1000
RESULT_SUMMARY_FILE = 'result_summary.txt'    

def log_result(cns_id, result):
    with open(RESULT_SUMMARY_FILE, 'a') as summary_file:
        summary_file.write(f"{cns_id},{result}\n")

def main_model(cns_id, tree, sequences, output_folder, pwm_db, thres_db, gc_dict):
    output_file = initialize_output_file(cns_id, output_folder + 'events/', '.events.csv')
    getEventsHistory(tree.root, cns_id, pwm_db, thres_db, sequences, output_file, gc_dict = gc_dict)

def neutral_main(cns_id, tree, sequences, n_simulations, pwm_db, thres_db, prob_mat, output_folder, min_events = MIN_EVENTS):
    events_folder =  output_folder + 'events/'
    output_file = initialize_output_file(cns_id, output_folder+'simulations/', '.simulation.csv')
    binding_tfs = get_binding_tfs(cns_id, events_folder, min_events)
    
    if len(binding_tfs) == 0:
        raise ValueError('No binding TFs')

    same_pos_dict = get_same_pos_dict(cns_id, events_folder)
    for tf in binding_tfs:
        getEventsProbability(cns_id, tree.root, sequences, prob_mat, n_simulations, tf, pwm_db, thres_db, same_pos_dict, output_file)
        


def main(cns_id, data_folder, output_folder, n_simulations = N_SIMULATIONS):
    try:
        step = "create_gc_dict"
        gc_dict = create_gc_dict(data_folder = data_folder)

        step = "importThresDB"
        thres_db = importThresDB(data_folder = data_folder)

        step = "createPWMdb"
        pwm_db = createPWMdb(data_folder = data_folder)

        step = 'get_prob_mat'
        prob_mat = get_prob_mat(data_folder)
        
        step = 'import_tree_and_sequences'
        tree, sequences = import_tree_and_sequences(cns_id, data_folder)
        
        step = "getEventsHistory"
        main_model(cns_id, tree, sequences, output_folder, pwm_db, thres_db, gc_dict)
        
        step = "neutral_main"
        neutral_main(cns_id, tree, sequences, n_simulations, pwm_db, thres_db, prob_mat, output_folder)

        log_result(cns_id, "completed successfully")
    except Exception as e:
        error_message = f"failed at {step}: {traceback.format_exc().splitlines()[-1]}"
        log_result(cns_id, error_message)
        print(f"Error with CNS ID {cns_id} at {step}: {e}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CNS data with specified options.")

    # Required argument
    parser.add_argument("--cns_id", required=True, type=str, help="ID of the CNS to process")

    # Optional arguments
    parser.add_argument("--n_simulations", default=N_SIMULATIONS, type=int, help="Number of simulations to run (default: 1000)")
    parser.add_argument("--data_folder", default=DATA_FOLDER, type=str, help="Path to the data folder (default: ../data/)")
    parser.add_argument("--output_folder", default=OUTPUT_FOLDER, type=str, help="Path to the output folder (default: ../output/)")
    args = parser.parse_args()

    main(
        cns_id=args.cns_id,
        data_folder=args.data_folder,
        output_folder=args.output_folder,
        n_simulations=args.n_simulations
    )
