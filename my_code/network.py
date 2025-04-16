from brian2 import *
import numpy as np


class NetworkObjects:
    """
    creates object where all neuron populations
    and synapses are stored, as well as the number
    of assemblies in the network, which must be fixed
    across different populations
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
    """
    standard Leaky-Integrate-and-Fire neuron model
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
            dv/dt = (curr_leak + curr_syn + curr_bg + curr_stim + curr_adapt)/mem_cap : volt (unless refractory)
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
        brian_obj.mem_cap = self.mem_cap.get_param()
        brian_obj.g_leak = self.g_leak.get_param()
        brian_obj.e_rest = self.e_rest.get_param()
        brian_obj.v_reset = self.v_reset.get_param()
        brian_obj.v_thres = self.v_thres.get_param()
        brian_obj.tau_refr = self.tau_refr.get_param()
        brian_obj.curr_bg = self.curr_bg.get_param()


class NeuronPopulationSettings:
    """
    object determining the properties of a brian2 NeuronGroup
    """
    def __init__(self,
                 name,
                 model,
                 n_cells,
                 adapt=None,
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
            adapt: output-driven adaptation model
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
        self.adapt = adapt
        self.synapses = synapses
        self.asb_flag = asb_flag
        self.asb_size = asb_size
        self.plot_color = plot_color
        self.raster_height = raster_height
        self.n_raster = n_raster
        self.brian_group = None
        self.monitors = []

        # add adaptation equations
        if self.adapt is not None:
            self.model.model_eqs += self.adapt.adapt_eqs
            self.model.reset_eqs += self.adapt.reset_eqs
        else:
            self.model.model_eqs += '''
            curr_adapt = 0 * amp : amp'''

        # add synaptic equations
        for synapse in self.synapses:
            self.model.curr_syn = self.model.curr_syn.replace(': amp', '')
            self.model.curr_syn += '+ curr_syn_' + synapse.pre_name + ' : amp'
            self.model.model_eqs += synapse.post_syn_eqs
        self.model.model_eqs += '''
            %s''' % self.model.curr_syn

    def create_brian_group(self, network):
        self.brian_group = NeuronGroup(self.n_cells,
                                       model=self.model.model_eqs,
                                       threshold=self.model.thres_eqs,
                                       reset=self.model.reset_eqs,
                                       refractory=self.model.refr_eqs,
                                       method='euler',
                                       name='pop_' + self.name)

        self.model.attr_params(self.brian_group)

        if self.adapt is not None:
            self.adapt.attr_params(self.brian_group)

        for synapse in self.synapses:
            synapse.attr_params(self.brian_group)

        network.add(self.brian_group)

    def create_firing_monitors(self, monitors, network, settings, max_record):
        net_objects = settings.net_objects

        spm_all_name = 'spm_' + self.name
        if spm_all_name not in network:
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

        # if there are assemblies:
        if self.asb_flag:

            # assembly spike monitors:
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

            # background neurons' spike monitor:
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

    def create_v_monitors(self, monitors, network, settings, max_record):

        sim_params = settings.sim_params
        net_objects = settings.net_objects
        sim_seed = sim_params['sim_seed'].get_param()

        stm_v_name = 'stm_' + self.name + '_v'
        if stm_v_name not in network:
            # record generally from cells
            rec_gen_idx = np.random.default_rng(sim_seed).choice(self.brian_group.N, size=max_record, replace=False)

            stm_v = StateMonitor(self.brian_group, 'v', record=rec_gen_idx, name=stm_v_name)
            network.add(stm_v)
        monitors += [stm_v_name]

        # if there are assemblies
        if self.asb_flag:
            n_asb = net_objects.n_asb

            # record from assemblies:
            for i in range(n_asb):

                stm_asb_i_v_name = 'stm_' + self.name + '_asb_' + str(i + 1) + '_v'
                if stm_asb_i_v_name not in network:
                    if self.asb_size > max_record:
                        rec_asb_idx = np.random.default_rng(sim_seed).choice(self.asb_size,
                                                                              size=max_record, replace=False)
                    else:
                        rec_asb_idx = True

                    stm_asb_i_v = StateMonitor(self.brian_group[i * self.asb_size: (i + 1) * self.asb_size], 'v',
                                               record=rec_asb_idx,
                                               name=stm_asb_i_v_name)
                    network.add(stm_asb_i_v)
                monitors += [stm_asb_i_v_name]

            # record from background neurons:
            if self.brian_group.N > n_asb * self.asb_size:

                stm_bg_v_name = 'stm_' + self.name + '_bg_v'
                if stm_bg_v_name not in network:
                    if self.brian_group.N - n_asb * self.asb_size > max_record:
                        rec_bg_idx = np.random.default_rng(sim_seed).choice(self.brian_group.N - n_asb * self.asb_size,
                                                                             size=max_record, replace=False)
                    else:
                        rec_bg_idx = True

                    stm_bg_v = StateMonitor(self.brian_group[n_asb * self.asb_size:], 'v',
                                            record=rec_bg_idx,
                                            name=stm_bg_v_name)
                    network.add(stm_bg_v)
                monitors += [stm_bg_v_name]

    def create_single_v_monitor(self, monitors, network, settings, asb_idx, max_record):

        sim_params = settings.sim_params
        sim_seed = sim_params['sim_seed'].get_param()

        n_asb = settings.net_objects.n_asb
        if asb_idx == 0:
            if self.brian_group.N > n_asb * self.asb_size:
                stm_v_name = 'stm_' + self.name + '_bg_v'
                if stm_v_name not in network:
                    if self.brian_group.N - n_asb * self.asb_size > max_record:
                        rec_bg_idx = np.random.default_rng(sim_seed).choice(self.brian_group.N - n_asb * self.asb_size,
                                                                            size=max_record, replace=False)
                    else:
                        rec_bg_idx = True

                    stm_v = StateMonitor(self.brian_group[n_asb * self.asb_size:], 'v',
                                            record=rec_bg_idx,
                                            name=stm_v_name)
                    network.add(stm_v)
                monitors += [stm_v_name]
        else:
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


    def create_all_current_monitors(self, monitors, network, settings, max_record):
        sim_params = settings.sim_params
        net_objects = settings.net_objects
        sim_seed = sim_params['sim_seed'].get_param()

        # record generally from cells
        rec_gen_idx = np.random.default_rng(sim_seed).choice(self.brian_group.N, size=max_record, replace=False)

        for pop_name in net_objects.pop_settings:
            if hasattr(self.brian_group, 'curr_syn_' + pop_name):
                stm_curr_syn_name = 'stm_%s_curr_syn_%s' % (self.name, pop_name)
                if stm_curr_syn_name not in network:
                    stm_curr = StateMonitor(self.brian_group,
                                            'curr_syn_' + pop_name,
                                            record=rec_gen_idx,
                                            name=stm_curr_syn_name)
                    network.add(stm_curr)
                monitors += [stm_curr_syn_name]

        # if there are assemblies
        if self.asb_flag:
            n_asb = net_objects.n_asb

            # record from assemblies:
            for i in range(n_asb):
                if self.asb_size > max_record:
                    rec_asb_idx = np.random.default_rng(sim_seed).choice(self.asb_size,
                                                                          size=max_record, replace=False)
                else:
                    rec_asb_idx = True

                for pop_name in net_objects.pop_settings:
                    if hasattr(self.brian_group, 'curr_syn_' + pop_name):
                        stm_asb_i_curr_name = 'stm_%s_asb_%d_curr_syn_%s' % (self.name, i + 1, pop_name)
                        if stm_asb_i_curr_name not in network:
                            stm_asb_i_curr = StateMonitor(self.brian_group[i * self.asb_size: (i + 1) * self.asb_size],
                                                          'curr_syn_' + pop_name,
                                                          record=rec_asb_idx,
                                                          name=stm_asb_i_curr_name)
                            network.add(stm_asb_i_curr)
                        monitors += [stm_asb_i_curr_name]

            # record from background neurons:
            if self.brian_group.N > n_asb * self.asb_size:

                if self.brian_group.N - n_asb * self.asb_size > max_record:
                    rec_bg_idx = np.random.default_rng(sim_seed).choice(self.brian_group.N - n_asb * self.asb_size,
                                                                         size=max_record, replace=False)
                else:
                    rec_bg_idx = True

                for pop_name in net_objects.pop_settings:
                    if hasattr(self.brian_group, 'curr_syn_' + pop_name):
                        stm_bg_curr_name = 'stm_%s_bg_curr_syn_%s' % (self.name, pop_name)
                        if stm_bg_curr_name not in network:
                            stm_bg_curr = StateMonitor(self.brian_group[n_asb * self.asb_size:],
                                                      'curr_syn_' + pop_name,
                                                      record=rec_bg_idx,
                                                      name=stm_bg_curr_name)
                            network.add(stm_bg_curr)
                        monitors += [stm_bg_curr_name]

    def create_single_current_monitor(self, monitors, network, settings, pre_pop, max_record):
        sim_params = settings.sim_params
        net_objects = settings.net_objects
        sim_seed = sim_params['sim_seed'].get_param()

        # record generally from cells
        rec_gen_idx = np.random.default_rng(sim_seed).choice(self.brian_group.N, size=max_record, replace=False)

        if hasattr(self.brian_group, 'curr_syn_' + pre_pop):
            stm_curr_name = 'stm_%s_curr_syn_%s' % (self.name, pre_pop)
            if stm_curr_name not in network:
                stm_curr = StateMonitor(self.brian_group,
                                        'curr_syn_' + pre_pop,
                                        record=rec_gen_idx,
                                        name=stm_curr_name)
                network.add(stm_curr)
            monitors += [stm_curr_name]

    def create_weight_monitors(self, monitors, network, settings, pre_pop, max_record):
        sim_params = settings.sim_params
        net_objects = settings.net_objects
        sim_seed = sim_params['sim_seed'].get_param()

        n_asb = settings.net_objects.n_asb

        if ('syn_' + self.name + pre_pop) in network:
            syn_obj = network['syn_' + self.name + pre_pop]

            # bg synapses
            syn_indices = np.where((syn_obj.j[:] >= n_asb * self.asb_size))[0]

            if len(syn_indices) > 0:
                rec_gen_idx = np.random.default_rng(sim_seed).choice(syn_indices,
                                                                     size=min(len(syn_indices), max_record),
                                                                     replace=False)
                stm_g_name = 'stm_g_%s_out' % (self.name + pre_pop)
                if stm_g_name not in network:
                    stm_g = StateMonitor(syn_obj, 'g_' + self.name + pre_pop, record=rec_gen_idx, name=stm_g_name)
                    network.add(stm_g)
                    monitors += [stm_g_name]

            # asb synapses
            for asb_idx in range(n_asb):
                syn_indices = np.where((syn_obj.j[:] >= asb_idx * self.asb_size) &
                                       (syn_obj.j[:] < (asb_idx + 1) * self.asb_size))[0]
                if len(syn_indices) > 0:
                    rec_gen_idx = np.random.default_rng(sim_seed).choice(syn_indices,
                                                                         size=min(len(syn_indices), max_record),
                                                                         replace=False)
                    stm_g_name = 'stm_g_%s_asb_%d' % (self.name + pre_pop, asb_idx + 1)
                    if stm_g_name not in network:
                        stm_g = StateMonitor(syn_obj, 'g_' + self.name + pre_pop, record=rec_gen_idx, name=stm_g_name)
                        network.add(stm_g)
                        monitors += [stm_g_name]
        else:

            # bg synapses
            syn_obj_name = 'syn_' + self.name + pre_pop + '_bg'
            if syn_obj_name in network:
                syn_obj = network[syn_obj_name]
                n_syn = syn_obj.N[:]
                if n_syn > 0:
                    rec_gen_idx = np.random.default_rng(sim_seed).choice(n_syn,
                                                                         size=min(n_syn, max_record),
                                                                         replace=False)
                    stm_g_name = 'stm_g_%s_out' % (self.name + pre_pop)
                    if stm_g_name not in network:
                        stm_g = StateMonitor(syn_obj, 'g_' + self.name + pre_pop, record=rec_gen_idx, name=stm_g_name)
                        network.add(stm_g)
                        monitors += [stm_g_name]

            # asb synapses
            for asb_idx in range(n_asb):
                syn_obj_name = 'syn_' + self.name + pre_pop + '_rc_%d' % (asb_idx + 1)
                if syn_obj_name in network:
                    syn_obj = network[syn_obj_name]
                    n_syn = syn_obj.N[:]
                    if n_syn > 0:
                        rec_gen_idx = np.random.default_rng(sim_seed).choice(n_syn,
                                                                             size=min(n_syn, max_record),
                                                                             replace=False)
                        stm_g_name = 'stm_g_%s_asb_%d' % (self.name + pre_pop, asb_idx + 1)
                        if stm_g_name not in network:
                            stm_g = StateMonitor(syn_obj, 'g_' + self.name + pre_pop, record=rec_gen_idx, name=stm_g_name)
                            network.add(stm_g)
                            monitors += [stm_g_name]

class PoissonPopulationSettings:
    """
    object determining the properties of a brian2 PoissonGroup,
    instead of creating neurons, only their output spikes are
    created, each following a Poisson process
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
        self.brian_group = PoissonGroup(self.n_cells,
                                        rates=self.rate,
                                        name='pop_' + self.name)

        network.add(self.brian_group)

    def create_firing_monitors(self, monitors, network, settings, max_record):

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
        init_state: string specifying in which state to initialise the network
        sim_seed: simulation random seed
    """

    sim_seed = sim_params['sim_seed'].get_param()
    init_state = sim_params['init_state'].get_param()

    if ((init_state != 'rand') and (init_state != 'uniform') and
            (init_state != 'normal') and (init_state != 'non-swr') and (init_state != 'swr')):
        raise ValueError('invalid init state')

    np.random.seed(sim_seed)

    pop_str_list = list(net_objects.pop_settings.keys())

    for pop_str in pop_str_list:
        pop_obj = None
        if 'pop_' + pop_str in built_network:
            pop_obj = built_network['pop_' + pop_str]

        if init_state == 'rand':
            if isinstance(pop_obj, NeuronGroup):

                # membrane potentials
                pop_obj.v = pop_obj.v_reset + (pop_obj.v_thres - pop_obj.v_reset) * np.random.rand(pop_obj.N)

                # synaptic conductances
                for presyn_str in pop_str_list:
                    if hasattr(pop_obj, 'g_' + presyn_str):
                        setattr(pop_obj, 'g_' + presyn_str, 0.1 * nS * np.random.rand(pop_obj.N))

        elif init_state == 'uniform':
            if isinstance(pop_obj, NeuronGroup):
                width = 6
                pop_obj.v = np.random.uniform(pop_obj.v_thres / mV - width, pop_obj.v_thres / mV - 0.1, size=pop_obj.N) * mV

        elif init_state == 'normal':
            if isinstance(pop_obj, NeuronGroup):
                sigma = sim_params['init_normal_sigma'].get_param() / mV
                mean = sim_params['init_normal_mean'].get_param() / mV
                gen_v_distr = np.random.normal(mean, sigma, size=pop_obj.N)

                # those that would start above threshold, will start at reset (to not create spikes at t=0)
                thres = pop_obj.v_thres / mV
                reset = pop_obj.v_reset / mV
                gen_v_distr[gen_v_distr > thres] = reset[gen_v_distr > thres]
                pop_obj.v = gen_v_distr * mV

        elif init_state == 'non-swr':
            if isinstance(pop_obj, NeuronGroup):
                if pop_obj.name[-1] == 'p' or pop_obj.name[-1] == 'b':
                    pop_obj.v = pop_obj.v_reset

                elif pop_obj.name[-1] == 'a':
                    pop_obj.v = pop_obj.v_reset + (pop_obj.v_thres - pop_obj.v_reset) * np.random.rand(pop_obj.N)

        elif init_state == 'swr':
            if isinstance(pop_obj, NeuronGroup):
                if pop_obj.name[-1] == 'a':
                    pop_obj.v = pop_obj.v_reset

                elif pop_obj.name[-1] == 'p' or pop_obj.name[-1] == 'b':
                    pop_obj.v = pop_obj.v_reset + (pop_obj.v_thres - pop_obj.v_reset) * np.random.rand(pop_obj.N)


class ChangeAttribute:
    """
    object specifying the change of a given network attribute
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


class VoltageStimulus:
    """
    object specifying a stimulation current injected to a target in the network
    """
    def __init__(self, target, time, frac=1.0, spread=0.0 * ms,
                 asb=0, shade=True, random_seed=1, type='gauss'):
        """
        Args:
            target: target population
            ...
        """
        self.target = target
        self.time = time
        self.frac = frac
        self.spread = spread
        self.asb = asb
        self.shade = shade
        self.random_seed = random_seed
        self.type = type
        self.applied = None
        self.checked_replay = False

        self.brian_pop = None

        if self.asb != 0:
            n_cells = self.target.asb_size
        else:
            n_cells = self.target.n_cells

        n_spikes = int(n_cells * frac)

        sim_dt = defaultclock.dt
        precision = len(str(sim_dt / second).split('.')[1])
        rng = np.random.default_rng(seed=self.random_seed)
        jump_cells = rng.choice(np.arange(n_cells), size=n_spikes, replace=False)

        if type == 'uniform':
            jump_times = np.round(rng.uniform(low=self.time / second, high=(self.time + self.spread) / second,
                                              size=n_spikes),
                                  precision)
        else:
            jump_times = np.round(rng.normal(loc=(self.time + 3 * self.spread) / second, scale=self.spread / second,
                                             size=n_spikes),
                                  precision)
        self.jump_dict = {}
        for time, cell in zip(jump_times, jump_cells):
            if time in self.jump_dict:
                self.jump_dict[time].append(cell)
            else:
                self.jump_dict[time] = [cell]

        # Create NetworkOperation object
        self.operation = NetworkOperation(self.jump_voltage, dt=defaultclock.dt)


    def jump_voltage(self):
        stim_asb = self.asb

        if stim_asb > 0:
            n_i_asb = self.target.asb_size
            stim_idx0 = (stim_asb - 1) * n_i_asb
        else:
            stim_idx0 = 0

        jump_times = np.array(list(self.jump_dict.keys()))
        sim_dt = defaultclock.dt
        precision = len(str(sim_dt / second).split('.')[1])
        curr_time = np.round(defaultclock.t / second, precision)

        if curr_time in jump_times:
            stim_cells = self.jump_dict[curr_time]

            for cell in stim_cells:
                self.brian_pop[stim_idx0 + cell].v = self.target.model.v_thres.get_param() + 0.1 * mV

