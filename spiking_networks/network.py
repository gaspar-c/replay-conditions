"""
network.py
Utilities to create Brian2 network objects and population settings.

This module defines wrappers around Brian2 populations and
monitors used by the simulation scripts. It centralises population
configuration (neuron models, synapse wiring) and helper
objects that represent events or attribute changes during a simulation.

Key classes:
- `NetworkObjects` - container for population/synapse settings and assembly count
- `LIFModel` - parameterised leaky integrate-and-fire neuron model
- `NeuronPopulationSettings` - describes a NeuronGroup and its monitors
- `PoissonPopulationSettings` - describes a PoissonGroup and monitors
- `ChangeAttribute`, `TriggerSpikes` - event objects used during simulations
"""
import numpy as np
from brian2 import NeuronGroup, PoissonGroup, SpikeMonitor, PopulationRateMonitor, StateMonitor, nS, mV


class NetworkObjects:
    """Container for population and synapse settings.

    Stores mappings of population and synapse settings by name and the
    number of assemblies (`n_asb`) in the network. The assembly count is
    expected to be consistent across populations that use assemblies.
    """
    def __init__(self):
        self.pop_settings = {}
        self.syn_settings = {}
        self.n_asb = None

    def add_to_pop(self, obj):
        self.pop_settings[obj.name] = obj

    def add_to_syn(self, obj):
        self.syn_settings[obj.name] = obj


class LIFModel:
    """Leaky integrate-and-fire neuron model definition.

    This object collects parameters and equations for a standard LIF
    neuron. `model_eqs`, `reset_eqs`, `thres_eqs` and `refr_eqs`
    attributes are used when creating a Brian2 `NeuronGroup`. An
    `attr_params(brian_obj)` helper  assigns parameter values to a
    created Brian2 object.
    """
    def __init__(self,
                 mem_cap,
                 g_leak,
                 e_rest,
                 v_reset,
                 v_thres,
                 tau_refr,
                 curr_bg
                 ):
        """
        Args:
            mem_cap: membrane capatitance
            g_leak: leak conductance
            e_rest: leak resting potential
            v_reset: reset potential
            v_thres: membrane threshold
            tau_refr: absolute refractory period
            curr_bg: constant background current
        """
        self.mem_cap = mem_cap
        self.g_leak = g_leak
        self.e_rest = e_rest
        self.v_reset = v_reset
        self.v_thres = v_thres
        self.tau_refr = tau_refr
        self.curr_bg = curr_bg

        self.model_eqs = '''
            dv/dt = (curr_leak + curr_syn + curr_bg + curr_stim)/mem_cap : volt (unless refractory)
            curr_leak = g_leak*(e_rest - v) : amp
            mem_cap : farad
            g_leak : siemens
            e_rest : volt
            v_thres : volt
            v_reset : volt
            tau_refr : second
            curr_bg : amp
            curr_stim : amp'''

        self.curr_syn = 'curr_syn = 0 * amp : amp'

        self.reset_eqs = '''
            v = v_reset
        '''

        self.thres_eqs = 'v > v_thres'
        self.refr_eqs = 'tau_refr'

    def attr_params(self, brian_obj):
        """Assign parameter values stored in this model to `brian_obj`.

        The `mem_cap`, `g_leak`, etc. members are expected to be Parameter
        containers providing a `get_param()` method.
        """
        brian_obj.mem_cap = self.mem_cap.get_param()
        brian_obj.g_leak = self.g_leak.get_param()
        brian_obj.e_rest = self.e_rest.get_param()
        brian_obj.v_reset = self.v_reset.get_param()
        brian_obj.v_thres = self.v_thres.get_param()
        brian_obj.tau_refr = self.tau_refr.get_param()
        brian_obj.curr_bg = self.curr_bg.get_param()


class NeuronPopulationSettings:
    """Configuration for a Brian2 `NeuronGroup` and its monitors.

    This class holds the neuron model, synapse descriptors and plotting
    preferences. It is responsible for assembling the combined model
    equations (including synaptic currents) and creating
    the Brian2 `NeuronGroup` and associated monitors.
    """
    def __init__(self,
                 name,
                 model,
                 n_cells,
                 synapses=[],
                 asb_flag=False,
                 asb_size=0,
                 plot_color='gray',
                 raster_height=1.0,
                 n_raster=250
                 ):
        """
        Args:
            name: name of neuron population
            model: neuron model
            n_cells: number of cells
            synapses: list of synaptic objects
            asb_flag: [True/False], determines whether population will contain assemblies
            asb_size: number of neurons in one assembly
            plot_color: default color for plots
            raster_height: proportional height of raster plot
            n_raster: number of neurons in raster plot
        """
        self.name = name
        self.model = model
        self.n_cells = n_cells
        self.synapses = synapses
        self.asb_flag = asb_flag
        self.asb_size = asb_size
        self.plot_color = plot_color
        self.raster_height = raster_height
        self.n_raster = n_raster
        self.brian_group = None
        self.monitors = []

        # add synaptic equations
        for synapse in self.synapses:
            self.model.curr_syn = self.model.curr_syn.replace(': amp', '')
            self.model.curr_syn += '+ curr_syn_' + synapse.pre_name + ' : amp'
            self.model.model_eqs += synapse.post_syn_eqs
        self.model.model_eqs += '''
            %s''' % self.model.curr_syn

    def create_brian_group(self, network):
        """Create and register the Brian2 `NeuronGroup` for this population.

        Parameters
        - network: the Brian2 `Network`-like container to add the created group to
        """
        self.brian_group = NeuronGroup(self.n_cells,
                                       model=self.model.model_eqs,
                                       threshold=self.model.thres_eqs,
                                       reset=self.model.reset_eqs,
                                       refractory=self.model.refr_eqs,
                                       method='euler',
                                       name='pop_' + self.name)

        # assign parameters for the model and any auxiliary components
        self.model.attr_params(self.brian_group)
        for synapse in self.synapses:
            synapse.attr_params(self.brian_group)

        network.add(self.brian_group)

    def create_firing_monitors(self, monitors, network, settings, max_record):
        """Create spike and rate monitors for this population.

        The function adds monitors to `network` and appends their names to
        the `monitors` list. It respects `max_record` to limit the number of
        recorded neurons for large populations. When the population contains
        assemblies (`asb_flag`), per-assembly monitors are created as well.
        """
        net_objects = settings.net_objects

        spm_all_name = 'spm_' + self.name
        if spm_all_name not in network:
            # limit recorded units when the population is large
            if self.n_cells > max_record:
                if self.asb_flag:
                    spm_all = SpikeMonitor(self.brian_group, record=False, name=spm_all_name)
                else:
                    spm_all = SpikeMonitor(self.brian_group[:max_record], name=spm_all_name)
            else:
                spm_all = SpikeMonitor(self.brian_group, name=spm_all_name)
            network.add(spm_all)
        monitors += [spm_all_name]

        rtm_all_name = 'rtm_' + self.name
        if rtm_all_name not in network:
            rtm_all = PopulationRateMonitor(self.brian_group, name=rtm_all_name)
            network.add(rtm_all)
        monitors += [rtm_all_name]

        # if there are assemblies: create per-assembly monitors and optional
        # monitors for neurons outside assembly groups
        if self.asb_flag:
            for i in range(net_objects.n_asb):
                spm_asb_i_name = 'spm_' + self.name + '_asb_' + str(i + 1)
                if spm_asb_i_name not in network:
                    spm_asb_i = SpikeMonitor(self.brian_group[i * self.asb_size:(i + 1) * self.asb_size],
                                             name=spm_asb_i_name)
                    network.add(spm_asb_i)
                monitors += [spm_asb_i_name]

                rtm_asb_i_name = 'rtm_' + self.name + '_asb_' + str(i + 1)
                if rtm_asb_i_name not in network:
                    rtm_asb_i = PopulationRateMonitor(self.brian_group[i * self.asb_size:(i + 1) * self.asb_size],
                                                      name=rtm_asb_i_name)
                    network.add(rtm_asb_i)
                monitors += [rtm_asb_i_name]

            # background neurons' spike monitor (neurons not belonging to assemblies)
            if self.brian_group.N > net_objects.n_asb * self.asb_size:
                end_rec = net_objects.n_asb * self.asb_size + max_record
                if end_rec > self.brian_group.N:
                    end_rec = -1

                spm_out_name = 'spm_' + self.name + '_out'
                if spm_out_name not in network:
                    spm_out = SpikeMonitor(self.brian_group[net_objects.n_asb * self.asb_size:end_rec],
                                           name=spm_out_name)
                    network.add(spm_out)
                monitors += [spm_out_name]

                rtm_out_name = 'rtm_' + self.name + '_out'
                if rtm_out_name not in network:
                    rtm_out = PopulationRateMonitor(self.brian_group[net_objects.n_asb * self.asb_size:end_rec],
                                                    name=rtm_out_name)
                    network.add(rtm_out)
                monitors += [rtm_out_name]

    def create_v_monitor(self, monitors, network, settings, asb_idx, max_record):
        """Create membrane potential StateMonitor for background or an assembly.

        - `asb_idx == 0` => create monitor for background neurons (those not in
          any assembly), sampling up to `max_record` units randomly.
        - `asb_idx > 0` => create monitor for the (asb_idx)th assembly, sampling
          up to `max_record` units.
        """

        sim_params = settings.sim_params
        sim_seed = sim_params['sim_seed'].get_param()

        n_asb = settings.net_objects.n_asb
        if asb_idx == 0:
            # background population outside the assembly blocks
            if self.brian_group.N > n_asb * self.asb_size:
                stm_v_name = 'stm_' + self.name + '_bg_v'
                if stm_v_name not in network:
                    if self.brian_group.N - n_asb * self.asb_size > max_record:
                        rec_bg_idx = np.random.default_rng(sim_seed).choice(
                            self.brian_group.N - n_asb * self.asb_size,
                            size=max_record, replace=False)
                    else:
                        rec_bg_idx = True

                    stm_v = StateMonitor(self.brian_group[n_asb * self.asb_size:], 'v',
                                          record=rec_bg_idx,
                                          name=stm_v_name)
                    network.add(stm_v)
                monitors += [stm_v_name]
        else:
            # monitor for a specific assembly (1-based index)
            stm_v_name = 'stm_' + self.name + '_asb_' + str(asb_idx) + '_v'
            if stm_v_name not in network:
                if self.asb_size > max_record:
                    rec_asb_idx = np.random.default_rng(sim_seed).choice(self.asb_size,
                                                                         size=max_record, replace=False)
                else:
                    rec_asb_idx = True

                stm_v = StateMonitor(self.brian_group[(asb_idx - 1) * self.asb_size: asb_idx * self.asb_size], 'v',
                                      record=rec_asb_idx,
                                      name=stm_v_name)
                network.add(stm_v)
            monitors += [stm_v_name]
    

class PoissonPopulationSettings:
    """Configuration for a Poisson spike population.

    This class describes a `PoissonGroup` which emits independent Poisson
    spike trains (one per cell) and provides methods to create the Brian2
    group and its standard monitors.
    """
    def __init__(self,
                 name,
                 n_cells,
                 rate,
                 plot_color='gray',
                 raster_height=1.0,
                 n_raster=250
                 ):
        """
        Args:
            name: name of neuron population
            n_cells: number of neurons
            rate: firing rate of each neuron
            plot_color: default color for plots
            raster_height: proportional height of raster plot
            n_raster: number of neurons in raster plot
        """
        self.name = name
        self.n_cells = n_cells
        self.rate = rate
        self.plot_color = plot_color
        self.raster_height = raster_height
        self.n_raster = n_raster
        self.brian_group = None

    def create_brian_group(self, network):
        """Create and register the Brian2 `PoissonGroup` for this population."""
        self.brian_group = PoissonGroup(self.n_cells,
                                        rates=self.rate,
                                        name='pop_' + self.name)

        network.add(self.brian_group)

    def create_firing_monitors(self, monitors, network, settings, max_record):
        """Create spike and rate monitors for this Poisson population.
        """
        spm_name = 'spm_' + self.name
        if spm_name not in network:
            if self.n_cells > max_record:
                spm_all = SpikeMonitor(self.brian_group[:max_record], name=spm_name)
            else:
                spm_all = SpikeMonitor(self.brian_group, name=spm_name)
            network.add(spm_all)
        monitors += [spm_name]

        rtm_name = 'rtm_' + self.name
        if rtm_name not in network:
            rtm_all = PopulationRateMonitor(self.brian_group, name=rtm_name)
            network.add(rtm_all)
        monitors += [rtm_name]


def init_network(built_network, net_objects, sim_params):
    """
    initialise network in a specified state

    Args:
        built_network: built brian network
        net_objects: objects used to build network
        init_state: string specifying with which random distribution to initialise the network
        sim_seed: simulation random seed
    """

    # Retrieve seed and requested initialization distribution
    sim_seed = sim_params['sim_seed'].get_param()
    init_state = sim_params['init_state'].get_param()

    if ((init_state != 'uniform') and (init_state != 'gaussian')):
        raise ValueError('invalid init state')

    # Use the requested seed to ensure reproducible initial conditions
    np.random.seed(sim_seed)

    pop_str_list = list(net_objects.pop_settings.keys())

    # Initialise each population present in the built network
    for pop_str in pop_str_list:
        pop_obj = None
        if 'pop_' + pop_str in built_network:
            pop_obj = built_network['pop_' + pop_str]

        # Uniform initialization: v between reset and threshold, small random conductances
        if init_state == 'uniform':
            if isinstance(pop_obj, NeuronGroup):
                pop_obj.v = pop_obj.v_reset + (pop_obj.v_thres - pop_obj.v_reset) * np.random.rand(pop_obj.N)

                # initialize per-presynaptic conductances if present
                for presyn_str in pop_str_list:
                    if hasattr(pop_obj, 'g_' + presyn_str):
                        setattr(pop_obj, 'g_' + presyn_str, 0.1 * nS * np.random.rand(pop_obj.N))

        # Gaussian initialization: sample from normal distribution but clip any
        # values that would start above threshold (to avoid spikes at t=0)
        elif init_state == 'gaussian':
            if isinstance(pop_obj, NeuronGroup):
                sigma = sim_params['init_gaussian_sigma'].get_param() / mV
                mean = sim_params['init_gaussian_mean'].get_param() / mV
                gen_v_distr = np.random.normal(mean, sigma, size=pop_obj.N)

                # those that would start above threshold, will start at reset
                thres = pop_obj.v_thres / mV
                reset = pop_obj.v_reset / mV
                gen_v_distr[gen_v_distr > thres] = reset[gen_v_distr > thres]
                pop_obj.v = gen_v_distr * mV


class ChangeAttribute:
    """Event describing a change to an attribute of a network object.

    Instances of this class are used by the simulation event system to apply
    attribute changes at a specified time.
    """
    def __init__(self, onset, target, attribute, value, subset=0):
        """
        Args:
            onset: time at which change happens
            target: target object
            attribute: attribute to be change
            value: new value for attribute
            subset: subset of population to target
        """
        self.onset = onset
        self.target = target
        self.attribute = attribute
        self.value = value
        self.subset = subset


class TriggerSpikes:
    """Event that forces membrane potentials above threshold, causing spikes.
    """

    def __init__(self, target, time, asb=0):
        """"
        Parameters
        - target: population settings object
        - time: time of trigger (Brian2 time unit)
        - asb: 0 for whole population, or 1-based assembly index to target
        """
        self.target = target
        self.time = time
        self.asb = asb
        self.checked_replay = False
        self.brian_pop = None  # set during simulation

        if self.asb != 0:
            n_cells = self.target.asb_size
            cell_idx0 = (self.asb - 1) * self.target.asb_size
        else:
            n_cells = self.target.n_cells
            cell_idx0 = 0

        self.cell_idxs = cell_idx0 + np.arange(n_cells)


    def make_spikes(self):
        """Force the selected neurons to spike by raising `v` above threshold."""
        for cell in self.cell_idxs:
            self.brian_pop[cell].v = self.target.model.v_thres.get_param() + 0.1 * mV
