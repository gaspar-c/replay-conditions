"""
model3_delta.py
Same as model3.py but with delta synapses for pp connections.
"""
import copy
from brian2 import Network, Hz, mV, nS, ms, second, pF, pA
from general_code.aux_functions import xprint
from spiking_networks.simulations import SimElement, run_network_sim
from general_code.parameters import Parameter, initialize_params
from spiking_networks.tests import NetworkTests, TestReplay
from spiking_networks.network import TriggerSpikes
from spiking_networks.plot_spiking_trace import PlotRaster, PlotPopRate, PlotV
from spiking_networks import network as net, synapses as syn, connectivities as conn


def run_simulation(options):

    sim_params = {
                  'n_stims': Parameter(1),
                  'conn_seed': Parameter(1),
                  'n_p': Parameter(5000),
                  'm_p': Parameter(500),
                  'n_e': Parameter(5000),
                  'p_pe': Parameter(0.01),
                  'n_asb': Parameter(10),
                  'p_bg': Parameter(0.00),
                  'p_rc': Parameter(0.08),
                  'p_ff': Parameter(0.08),
                  'rate_ext': Parameter(50, Hz),
                  'syn_weight_pe': Parameter(0.06, mV),
                  'syn_weight_pp': Parameter(0.05, mV),
                  'tau_ref': Parameter(1, ms),
                  'tau_l_pp': Parameter(3, ms),
                  'v_time': Parameter(16),
                  'conn_fixed': Parameter(False),
                  'mem_cap': Parameter(200, pF),
                  'g_leak': Parameter(10, nS),
                  'curr_bg': Parameter(58, pA)
    }

    log, sim_params, plot_params = initialize_params(options, sim_params)
    network_objects = net.NetworkObjects()

    # Build Brian2 spiking neural network model
    xprint('================ Building Network ================', log)

    # Define external 'e' Poisson population
    pop_e_sett = net.PoissonPopulationSettings(name='e',
                                               n_cells=sim_params['n_e'].get_param(),
                                               rate=sim_params['rate_ext'].get_param(),
                                               plot_color='black')
    network_objects.add_to_pop(pop_e_sett)

    # Create standard LIF neuron model
    std_lif = net.LIFModel(mem_cap=sim_params['mem_cap'],
                           g_leak=sim_params['g_leak'],
                           e_rest=Parameter(-60, mV),
                           v_reset=Parameter(-60, mV),
                           v_thres=Parameter(-50, mV),
                           tau_refr=sim_params['tau_ref'],
                           curr_bg=sim_params['curr_bg'])

    # Define population 'p' (principal cells)
    pop_p_sett = net.NeuronPopulationSettings(name='p',
                                              model=copy.copy(std_lif),
                                              n_cells=sim_params['n_p'].get_param(),
                                              asb_flag=True,
                                              asb_size=sim_params['m_p'].get_param(),
                                              synapses=[
                                                  syn.PostSynapseNone(pre_name='p'),
                                                  syn.PostSynapseNone(pre_name='e')],
                                              plot_color=plot_params['p_color'].get_param(),
                                              raster_height=4.0)
    network_objects.add_to_pop(pop_p_sett)

    # Create Brian2 network and add populations
    brian_network = Network()
    pop_p_sett.create_brian_group(brian_network)
    pop_e_sett.create_brian_group(brian_network)

    # Define and connect synaptic pathways
    network_objects.n_asb = sim_params['n_asb'].get_param()

    syn_pp_sett = conn.ConnMorphSequence(name='pp',
                                         pre_name='p', post_name='p',
                                         conn_seed=sim_params['conn_seed'].get_param(),
                                         prob_bg=sim_params['p_bg'].get_param(),
                                         prob_rc=sim_params['p_rc'].get_param(),
                                         prob_ff=sim_params['p_ff'].get_param(),
                                         conn_fixed=sim_params['conn_fixed'].get_param(),
                                         syn_model=syn.SynapseCurrentDirac(
                                             name='pp',
                                             weight=sim_params['syn_weight_pp'],
                                             latency=sim_params['tau_l_pp'])
                                         )
    syn_pp_sett.connect(network_objects, brian_network, log=log)
    network_objects.add_to_syn(syn_pp_sett)

    syn_pe_sett = conn.ConnRandomToSequence(name='pe',
                                            pre_name='e', post_name='p',
                                            conn_seed=sim_params['conn_seed'].get_param(),
                                            prob=sim_params['p_pe'].get_param(),
                                            conn_fixed=False,
                                            syn_model=syn.SynapseCurrentDirac(
                                                name='pe',
                                                weight=sim_params['syn_weight_pe'],
                                                latency=Parameter(1, ms))
                                           )
    syn_pe_sett.connect(network_objects, brian_network, log=log)
    network_objects.add_to_syn(syn_pe_sett)

    xprint('================ Network Built ================', log)

    """ SIMULATION EVENTS AND TESTS """

    events = []
    monitors = []

    # Set up replay test events
    wait_time = 1 * second
    inter_stim_time = 1 * second
    n_stims = sim_params['n_stims'].get_param()
    for stim_idx in range(n_stims):
        stim_time = wait_time + stim_idx * inter_stim_time

        # Stimulate assembly 1
        events.append(TriggerSpikes(target=pop_p_sett, time=stim_time, asb=1))

        # Test replay success
        events.append(NetworkTests(monitors=monitors,
                                   start=stim_time - 50 * ms, stop=stim_time + 400 * ms,
                                   max_record=sim_params['max_record'].get_param(),
                                   test_list=[
                                      TestReplay(pop=pop_p_sett, filter_width=2 * ms,
                                                  detect_range=80 * ms, min_height=30 * Hz,
                                                  min_dist=sim_params['tau_l_pp'].get_param())
                                   ],
                                   plot_list=[]
                                   ))

        # Plot replay results
        if options['output_plots']:
            events.append(NetworkTests(monitors=monitors,
                                       start=stim_time - 10 * ms, stop=stim_time + 120 * ms,
                                       max_record=sim_params['max_record'].get_param(),
                                       test_list=[
                                       ],
                                       plot_list=[
                                          PlotRaster(pops=[pop_p_sett]),
                                          PlotPopRate(pops=[pop_p_sett], filter_width=2 * ms, y_max=[250]),
                                          PlotV(pop=pop_p_sett, asb=[10],
                                                title=r''),
                                       ],
                                       time_bar=50*ms
                                       ))

    # Run network simulation
    sim_settings = SimElement(options=options,
                              brian_net=brian_network,
                              net_objects=network_objects,
                              sim_params=sim_params,
                              plot_params=plot_params,
                              events=events,
                              monitors=monitors,
                              log=log)
    run_network_sim(sim_settings)
