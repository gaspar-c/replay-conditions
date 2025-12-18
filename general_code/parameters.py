"""
parameters.py
Parameter utilities and default parameter sets used by the simulations.

This module provides a `Parameter` container used throughout the
codebase to hold values together with a preferred unit and usage flag.
It also contains helpers to log used parameter values
"""

import random
import textwrap
from brian2 import Quantity, NeuronGroup, PoissonGroup, Synapses, ms, cm
from general_code.aux_functions import xprint


class Parameter:
    """Container for a simulation parameter value and preferred unit.

    Instances hold a `quantity` (a Brian2-compatible value), a
    `pref_unit` used for printing and a `used` flag that is set when
    the parameter is read via `get_param()`.
    """

    def __init__(self, val, pref_unit=1):
        """Initialise the Parameter.

        Parameters
        - val: raw value (should not already be a `brian2.Quantity`)
        - pref_unit: preferred unit (Brian2 unit or 1 for unitless)
        """

        # store the numeric value as a Quantity using the preferred unit
        if type(val) is Quantity:
            print('ERROR: Parameter value should not have units')

        if type(val) is bool:
            # booleans are stored as-is (no unit scaling)
            self.quantity = val
        else:
            self.quantity = val * pref_unit

        # preferred unit for string formatting
        self.pref_unit = pref_unit
        # flag to be set when the parameter is used
        self.used = False

    def get_param(self):
        """Return the stored quantity and mark the parameter as used."""
        self.used = True
        return self.quantity

    def get_str(self):
        """
        Return parameter quantity as string 'val * unit' and mark parameter as used.
        """
        self.used = True

        pref_unit_str = str(self.pref_unit)
        if '1. ' in pref_unit_str:
            pref_unit_str = pref_unit_str[3:]

        output_string = str(self.quantity / self.pref_unit) + ' * ' + pref_unit_str
        return output_string


def print_attr_deep(name, val):
    """Format a single attribute for printing.

    Handles `Parameter` instances by requesting a string representation and
    recursively prints nested objects that expose `__dict__`.
    """
    text_out = ''
    if type(val) is Parameter:
        val = val.get_str()
    text_out += '%s: %s\n' % (name, val)

    # for complex types attempt to recurse into attributes
    if not (isinstance(val, (int, float, str, bool, Parameter, NeuronGroup, PoissonGroup, Synapses)) or (val is None)):
        if hasattr(val, '__dict__'):
            text_out += textwrap.indent(print_attr(val), '\t')
        else:
            text_out += '\t%s: ERROR: Unknown Type\n' % name

    return text_out


def print_attr(obj):
    """Return a formatted string listing attributes of `obj`.

    Lists are expanded so that each element is printed individually using
    `print_attr_deep`.
    """
    text_out = ''
    for attr_name in vars(obj):
        attr_val = getattr(obj, attr_name)
        if type(attr_val) is list:
            for attr_i in attr_val:
                text_out += print_attr_deep(attr_name, attr_i)
        else:
            text_out += print_attr_deep(attr_name, attr_val)

    return text_out


def get_dft_sim_params():
    """Return a dictionary with default simulation parameters.
    """
    dft_sim_params = {
        'sim_dt': Parameter(0.1, ms),
        'sim_seed': Parameter(100),
        'max_record': Parameter(1000),
        'init_state': Parameter('uniform')
    }

    return dft_sim_params


def get_dft_plot_params():
    """Return a dictionary with default plotting parameters.
    """

    dft_plot_params = {
        'fig_height': Parameter(8, cm),
        'fig_width': Parameter(6, cm),
        'spine_width': Parameter(1.0),
        'p_color': Parameter('#ef3b53'),
        'b_color': Parameter('dodgerblue'),
        'text_font': Parameter(9),
        'hide_time_ax': Parameter(True)
    }

    return dft_plot_params


def load_param_overrides(target_params, param_overrides, log=None):
    """Override entries in `target_params` with values from `param_overrides`.

    `param_overrides` may contain raw values, tuples `(val, unit)` or already
    constructed `Parameter` instances. Values are converted to `Parameter`
    objects stored in `target_params` and the resulting assignment is logged via
    `xprint` when `log` is provided.
    """
    for item in param_overrides.keys():
        param_k = param_overrides[item]
        if type(param_k) is tuple:
            target_params[item] = Parameter(*param_k)
        elif isinstance(param_k, Parameter):
            target_params[item] = param_k
        else:
            target_params[item] = Parameter(param_k)
        xprint('\t %s = %s' % (item, target_params[item].get_param()), log)


def initialize_params(options, specified_params):
    """
    initializes parameters for an individual simulation
    """

    # import default parameters
    sim_params = get_dft_sim_params()
    plot_params = get_dft_plot_params()

    # log command line:
    log = (options['output_dir'] + options['group_label'] + '/' + 'sim' + str(options['sim_idx']) +
           '.' + options['group_param_array_str'] + '.log')

    # load specified parameters to override defaults:
    load_param_overrides(sim_params, specified_params)

    # load from group param arrays:
    if 'group_param_overrides' in options:
        xprint('Simulation parameters:', log)
        load_param_overrides(sim_params, options['group_param_overrides'], log)

    # initialise python seed:
    sim_seed = sim_params['sim_seed'].get_param()
    random.seed(sim_seed)

    return log, sim_params, plot_params