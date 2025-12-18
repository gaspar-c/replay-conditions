"""
aux_functions.py
Helpers used across the simulations repository.
"""
import os
import glob
import builtins
import numpy as np
from brian2 import second


def xprint(string, logfile=None):
    """
    Print a message to stdout and optionally append to a logfile.

    Parameters
    - string: str - the message to print/write
    - logfile: str or None - path to logfile to append the message to
    """
    print(string)

    if logfile is not None:
        # Open in append mode so repeated calls accumulate logs
        logger = open(logfile, 'a')
        logger.write(string + '\n')
        logger.close()


def bool_from_str(str_array):
    """
    Convert an array of string values ('True'/'False') to a boolean array.

    Args:
        str_array: Array of strings ('True' or 'False').

    Returns:
        bool_array: Array of booleans.
    """

    bool_array = np.zeros_like(str_array, dtype=bool)
    for k in range(len(str_array)):
        if str_array[k] == 'True':
            bool_array[k] = True
        elif str_array[k] == 'False':
            bool_array[k] = False
        else:
            print('ERROR str not True/False!')
            exit()

    return bool_array


def float_from_str(str_array, unit=1):
    """
    Convert an array of string values to floats, optionally applying a unit multiplier.

    Args:
        str_array: Array of strings representing numbers.
        unit: Multiplier to apply to each value (default 1).

    Returns:
        val_array: Array of floats.
    """
    np_str = np.array(str_array, dtype=str)
    star_position = np.char.find(np_str, '*')
    val_array = np.zeros_like(np_str, dtype=float) * unit

    for k in range(len(str_array)):
        if star_position[k] > 0:
            val_array[k] = float(np_str[k][:star_position[k]])
        else:
            val_array[k] = float(np_str[k])

    return val_array


def seconds_to_hhmmss(seconds):
    """
    converts input in seconds to str in format 'hh:mm:ss'
    """
    string_out = ''
    hh = seconds // 3600
    if hh > 0:
        string_out += ('%d' % hh) + 'h '

    remainder = seconds % 3600
    mm = remainder // 60
    if mm > 0:
        string_out += ('%d' % mm) + 'm '

    ss = int(remainder % 60)
    string_out += ('%d' % ss) + 's '

    return string_out


def select_group_label_dir(output_subdir, pattern_prefix):
    """
    Interactively select one of the existing group label output directories.

    Args:
        output_subdir: name of the subdirectory under the current working
                       directory where outputs are stored (e.g. 'outputs').
        pattern_prefix: prefix of the group label directory pattern
                        (e.g. 'fig1_model1_D_').

    Returns:
        The full path of the selected directory.
    """
    output_dir_base = os.path.join(os.getcwd(), output_subdir)
    dir_pattern = os.path.join(output_dir_base, pattern_prefix + '*')
    group_label_dirs = sorted(glob.glob(dir_pattern))

    if not group_label_dirs:
        raise FileNotFoundError(
            f"No output directories matching '{pattern_prefix}*' found in {output_subdir}/"
        )

    print("Available output directories:")
    for idx, d in enumerate(group_label_dirs):
        print(f"{idx}: {d}")

    # use builtins.input to avoid clashes with brian2's input module
    selected_idx = int(builtins.input("Enter the index of the directory to use: "))
    if selected_idx < 0 or selected_idx >= len(group_label_dirs):
        raise IndexError("Selected index out of range.")

    return group_label_dirs[selected_idx]


def trim_brian_monitor(monitor, attr, attr_unit, t_start, t_stop):
    """
    trim a given brian monitor attribute to a given time range

    Args:
        monitor: brian monitor object
        attr: attribute of monitor (variable being measured)
        attr_unit: output unit of attribute
        t_start: start time of calculation
        t_stop: stop time of calculation

    Returns:
        time_array: trimmed time array
        attr_array: trimmed attribute array (unit-less)
    """

    time_array = np.array(monitor.t / second)[(monitor.t >= t_start) &
                                              (monitor.t <= t_stop)]

    attr_array = np.array(attr / attr_unit)[(monitor.t >= t_start) &
                                            (monitor.t <= t_stop)]

    return time_array, attr_array


def trim_brian_monitor_group(monitor, attr, attr_unit, t_start, t_stop):
    """
    trim a given brian monitor attribute to a given time range

    Args:
        monitor: brian monitor object
        attr: attribute of monitor (variable being measured)
        attr_unit: output unit of attribute
        t_start: start time of calculation
        t_stop: stop time of calculation

    Returns:
        time_array: trimmed time array
        attr_array: trimmed attribute array (unit-less)
    """

    time_array = np.array(monitor.t / second)[(monitor.t >= t_start) &
                                              (monitor.t <= t_stop)]

    attr_array = np.array(attr[:, (monitor.t >= t_start) &
                                  (monitor.t <= t_stop)] / attr_unit)

    return time_array, attr_array


def get_model_number(valid_models=None):
    """
    Prompt user to select a model number interactively.

    Args:
        valid_models (list or None): List of valid model numbers.
                                      If None, defaults to [1, 2, 3].

    Returns:
        int: Selected model number.
    """
    if valid_models is None:
        valid_models = [1, 2, 3]

    while True:
        try:
            model_num = int(builtins.input("Select model ({}): ".format(
                ', '.join(map(str, valid_models)))).strip())
            if model_num in valid_models:
                return model_num
            else:
                print(f"Invalid choice. Please enter one of: {valid_models}")
        except ValueError:
            print("Invalid input. Please enter a number.")



def get_model_shape(valid_models=None):
    """
    Prompt user to select a model shape interactively.

    Args:
        valid_models (list or None): List of valid model numbers.
                                     If None, defaults to ['gaussian', 'rectangle'].

    Returns:
        int: Selected model number.
    """
    if valid_models is None:
        valid_models = ['gaussian', 'rectangle']

    model_array_str = "{}".format(', '.join(map(str, valid_models))).strip().lower()

    while True:
        try:
            model_shape = builtins.input("Select model (%s): " % model_array_str).strip().lower()
            if model_shape in valid_models:
                return model_shape
            else:
                print(f"Invalid choice. Please enter one of: [%s]" % model_array_str)
        except ValueError:
            print("Invalid input. Please enter a valid model shape.")

