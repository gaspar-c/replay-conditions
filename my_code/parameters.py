from brian2 import *
from my_code.aux_functions import xprint
import textwrap


class Parameter:
    """
    simulation parameter
    """
    def __init__(self, val, pref_unit=1):
        """
        initialise parameter object, attributing it a quantity and preferred unit

        Args:
            val: parameter value
            pref_unit: preferred parameter unit
        """

        if type(val) is Quantity:
            print('ERROR: Parameter value should not have units')
        if type(val) is bool:
            self.quantity = val
        else:
            self.quantity = val * pref_unit
        self.pref_unit = pref_unit
        self.used = False

    def change_param(self, factor):
        """
        multiply parameter by some factor
        """
        self.quantity = self.quantity * factor

    def get_param(self):
        """
        output parameter quantity and mark parameter as used

        Returns:
            self.quantity: parameter quantity
        """

        self.used = True
        return self.quantity

    def get_str(self, use=True):
        """
        output parameter quantity as string in preferred unit and mark parameter as used

        Returns:
            output_string: string of parameter quantity in preferred unit
        """

        if use:
            self.used = True

        pref_unit_str = str(self.pref_unit)
        if '1. ' in pref_unit_str:
            pref_unit_str = pref_unit_str[3:]

        output_string = str(self.quantity / self.pref_unit) + ' * ' + pref_unit_str
        return output_string


def print_attr_deep(name, val):
    """
    create string with value of object attributes
    """
    text_out = ''
    if type(val) is Parameter:
        val = val.get_str()
    text_out += '%s: %s\n' % (name, val)

    if not (isinstance(val, (int, float, str, bool, Parameter, NeuronGroup, PoissonGroup, Synapses)) or (val is None)):
        if hasattr(val, '__dict__'):
            text_out += textwrap.indent(print_attr(val), '\t')
        else:
            text_out += '\t%s: ERROR: Unknown Type\n' % name

    return text_out


def print_attr(obj):
    """
    create string with attributes of a given object
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
    """
    dictionary with default simulation parameters
    """

    dft_sim_params = {
                       'sim_dt': Parameter(0.1, ms),
                       'sim_seed': Parameter(100),
                       'max_record': Parameter(1000),
                       'init_state': Parameter('rand')
                       }

    return dft_sim_params


def get_dft_plot_params():
    """
    dictionary with default plot parameters
    """

    dft_plot_params = {'fig_width': Parameter(20, cmeter),
                       'fig_height': Parameter(20, cmeter),
                       'p_color': Parameter('#ef3b53'),
                       'b_color': Parameter('dodgerblue'),
                       'a_color': Parameter('darkgreen'),
                       'param_font': Parameter(9),
                       'text_font': Parameter(9),
                       'spine_width': Parameter(1.0),
                       'hide_time_ax': Parameter(False)
                       }

    return dft_plot_params


def load_param_array(params_old, param_array, log=None):
    """
    override parameters in params_old with parameters in param_array
    """
    for item in param_array.keys():
        param_k = param_array[item]
        if type(param_k) is tuple:
            params_old[item] = Parameter(*param_k)
        elif isinstance(param_k, Parameter):
            params_old[item] = param_k
        else:
            params_old[item] = Parameter(param_k)
        xprint('\t %s = %s' % (item, params_old[item].get_param()), log)
