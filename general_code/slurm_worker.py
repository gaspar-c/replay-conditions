"""
slurm_worker.py
Entry point for individual Slurm array jobs submitted by run_sim_group.

Called automatically via sbatch — do not run directly.
Usage (internal): python -m general_code.slurm_worker <params_pkl> <module_path> <func_name>
"""
import os
import sys
import pickle
import importlib


def main():
    if len(sys.argv) < 4:
        print("Usage: python -m general_code.slurm_worker <params_pkl> <module_path> <func_name>")
        sys.exit(1)

    params_pkl  = sys.argv[1]
    module_path = sys.argv[2]
    func_name   = sys.argv[3]

    slurm_task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    if slurm_task_id is None:
        print("Error: SLURM_ARRAY_TASK_ID not set. This script must be run as a Slurm array job.")
        sys.exit(1)
    sim_idx = int(slurm_task_id)

    with open(params_pkl, 'rb') as f:
        data = pickle.load(f)
    group_options = data['group_options']
    group_params  = data['group_params']

    module     = importlib.import_module(module_path)
    run_single = getattr(module, func_name)

    from general_code.group_simulations import choose_from_group_params
    options_single = choose_from_group_params(group_options, group_params, sim_idx, n_cores=1)
    run_single(options_single)


if __name__ == '__main__':
    main()
