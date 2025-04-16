from brian2 import *
from my_code.parameters import Parameter


""" POST SYNAPTIC MODELS """


class PostSynapseCondSingExp:
    """
    models post-synaptic dynamics as an exponentially decaying conductance
    """
    def __init__(self,
                 pre_name,
                 e_nernst,
                 tau_d):
        """
        Args:
            pre_name: name of presynaptic population
            e_nernst: nernst potential
            tau_d: decay time constant
        """
        self.pre_name = pre_name
        self.e_nernst = e_nernst
        self.tau_d = tau_d

        self.post_syn_eqs = '''
            curr_syn_%s = g_%s * (e_%s - v): amp
            dg_%s / dt = -g_%s / tau_d_%s: siemens
            e_%s: volt
            tau_d_%s: second''' % (pre_name, pre_name, pre_name,
                                   pre_name, pre_name, pre_name,
                                   pre_name,
                                   pre_name)

    def attr_params(self, brian_obj):
        setattr(brian_obj, 'e_' + self.pre_name, self.e_nernst.get_param())
        setattr(brian_obj, 'tau_d_' + self.pre_name, self.tau_d.get_param())


class PostSynapseNone:
    """
    doesn't model post-synaptic dynamics, the synapse could
    work by directly shifting the membrane potential
    """
    def __init__(self,
                 pre_name):
        self.pre_name = pre_name

        self.post_syn_eqs = '''
            curr_syn_%s : amp
            ''' % pre_name

    def attr_params(self, syn_object):
        pass


""" SYNAPTIC MODELS WITHOUT PLASTICITY """


class SynapseCondNormal:
    """
    models synaptic dynamics as a jump in the synaptic conductance
    """
    def __init__(self,
                 name,
                 weight=None,
                 latency=None):
        """
        Args:
            name: name of synapse object
            weight: amount by which conductance jumps
            latency: synaptic latency
        """
        self.name = name
        self.weight = weight
        self.latency = latency

        str_pre = name[1]

        self.model_eqs = 'g_%s : siemens' % name
        self.on_pre_eqs = 'g_%s += g_%s' % (str_pre, name)
        self.on_post_eqs = ''

    def attr_params(self, syn_object):
        setattr(syn_object, 'g_%s' % self.name, self.weight.get_str())


class SynapseCurrentDirac:
    """
    models synaptic dynamics as a jump in the voltage (current Dirac-delta)
    """
    def __init__(self,
                 name,
                 weight=None,
                 latency=None):
        """
        Args:
            name: name of synapse object
            weight: amount by which the membrane potential jumps
            latency: synaptic latency
        """
        self.name = name
        self.weight = weight
        self.latency = latency
        str_pre = name[1]

        self.model_eqs = '''
            weight_%s : volt
        ''' % name

        self.on_pre_eqs = 'v += weight_%s' % name
        self.on_post_eqs = ''

    def attr_params(self, syn_object):
        setattr(syn_object, 'weight_' + self.name, self.weight.get_param())


""" SYNAPTIC MODELS WITH PLASTICITY """


class SynapseCondSTDP:
    """
    models synaptic dynamics as a jump in the conductance,
    but with a plasticity mechanism that changes the weight
    such that the post-synaptic population becomes balanced
    --- as done in Vogels-Sprekel (Science, 2011)
    """
    def __init__(self,
                 name,
                 g_init,
                 latency,
                 rho0,
                 tau,
                 g_max=Parameter(100, nS)):
        """
        Args:
            name: name of synapse object
            g_init: initial weight (conductance jump)
            latency: synaptic latency
            rho0: target firing rate for postsynaptic population
            tau: STDP time window
            g_max: maximum weight (conductance jump)
        """
        self.name = name
        self.g_init = g_init
        self.latency = latency
        self.rho0 = rho0
        self.tau = tau
        self.g_max = g_max

        str_pre = name[1]
        str_post = name[0]

        self.model_eqs = '''
                            g_%s : siemens
                            eta : 1
                            alpha = 2 * (%s) * (%s) : 1
                            dx_%s / dt = -x_%s / (%s) : 1 (event-driven)
                            dx_%s / dt = -x_%s / (%s) : 1 (event-driven)
                         ''' % (name,
                                self.rho0.get_str(), self.tau.get_str(),
                                str_pre, str_pre, self.tau.get_str(),
                                str_post, str_post, self.tau.get_str())

        self.on_pre_eqs = '''
                             x_%s += 1.
                             g_%s = clip(g_%s + (x_%s - alpha)*%s*eta, 0*nS, %s)
                             g_%s_post += g_%s
                          ''' % (str_pre,
                                 name, name, str_post, g_init.get_str(), self.g_max.get_str(),
                                 str_pre, name)

        self.on_post_eqs = '''
                              x_%s += 1.
                              g_%s = clip(g_%s + x_%s*%s*eta, 0*nS, %s)
                           ''' % (str_post,
                                  name, name, str_pre, g_init.get_str(), self.g_max.get_str())

    def attr_params(self, syn_object):
        setattr(syn_object, 'g_%s' % self.name, self.g_init.get_str())
        setattr(syn_object, 'eta', 0)
