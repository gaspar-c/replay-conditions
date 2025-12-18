"""
model3_no_leak.py
Similar to model3.py but without leak currents in neurons, and with fixed in-degree connectivity.
"""
import copy
from brian2 import Network, Hz, mV, nS, ms, second, pF, pA
from general_code.aux_functions import xprint
from spiking_networks.simulations import SimElement, run_network_sim
from general_code.parameters import Parameter, initialize_params
from spiking_networks.tests import NetworkTests, TestReplay
from spiking_networks import network as net, synapses as syn, connectivities as conn
from spiking_networks.network import TriggerSpikes



def run_simulation(options):

    sim_params = {
                  'conn_seed': Parameter(1),
                  'sim_seed': Parameter(1),
                  'init_state': Parameter('gaussian'),
                  'init_gaussian_mean': Parameter(-51.9, mV),
                  'init_gaussian_sigma': Parameter(0.5, mV),
                  'n_p': Parameter(5000),
                  'm_p': Parameter(500),
                  'n_asb': Parameter(10),
                  'p_bg': Parameter(0.00),
                  'p_rc': Parameter(0.00),
                  'p_ff': Parameter(0.12),
                  'g_pp': Parameter(0.10, nS),
                  'tau_l_pp': Parameter(1, ms)
    }

    log, sim_params, plot_params = initialize_params(options, sim_params)
    network_objects = net.NetworkObjects()

    # Build Brian2 spiking neural network model
    xprint('================ Building Network ================', log)

    # Create standard LIF neuron model
    std_lif = net.LIFModel(mem_cap=Parameter(200, pF),
                           g_leak=Parameter(0, nS),
                           e_rest=Parameter(-60, mV),
                           v_reset=Parameter(-60, mV),
                           v_thres=Parameter(-50, mV),
                           tau_refr=Parameter(1, ms),
                           curr_bg=Parameter(0, pA))

    # Define population 'p' (principal cells)
    pop_p_sett = net.NeuronPopulationSettings(name='p',
                                              model=copy.copy(std_lif),
                                              n_cells=sim_params['n_p'].get_param(),
                                              asb_flag=True,
                                              asb_size=sim_params['m_p'].get_param(),
                                              synapses=[
                                                  syn.PostSynapseCondSingExp(pre_name='p',
                                                                             e_nernst=Parameter(0, mV),
                                                                             tau_d=Parameter(2, ms))
                                                ],
                                              plot_color=plot_params['p_color'].get_param(),
                                              raster_height=4.0)
    network_objects.add_to_pop(pop_p_sett)

    # Create Brian2 network and add populations
    brian_network = Network()
    pop_p_sett.create_brian_group(brian_network)

    # Define and connect synaptic pathways
    network_objects.n_asb = sim_params['n_asb'].get_param()

    syn_pp_sett = conn.ConnMorphSequence(name='pp',
                                         pre_name='p', post_name='p',
                                         conn_seed=sim_params['conn_seed'].get_param(),
                                         prob_bg=sim_params['p_bg'].get_param(),
                                         prob_rc=sim_params['p_rc'].get_param(),
                                         prob_ff=sim_params['p_ff'].get_param(),
                                         conn_fixed=True,
                                         syn_model=syn.SynapseCondNormal(
                                             name='pp',
                                             weight=sim_params['g_pp'],
                                             latency=sim_params['tau_l_pp'])
                                         )
    syn_pp_sett.connect(network_objects, brian_network, log=log)
    network_objects.add_to_syn(syn_pp_sett)

    xprint('================ Network Built ================', log)

    """ SIMULATION EVENTS AND TESTS """

    events = []
    monitors = []


    # Set up replay test events
    wait_time = 1 * second
    stim_time = wait_time

    # Stimulate assembly 1
    events.append(TriggerSpikes(target=pop_p_sett, time=stim_time, asb=1))

    # Test replay success
    events.append(NetworkTests(monitors=monitors,
                                start=stim_time - 0.05 * second, stop=stim_time + 1 * second,
                                max_record=sim_params['max_record'].get_param(),
                                test_list=[
                                    TestReplay(pop=pop_p_sett, filter_width=7 * ms,
                                                min_height=15*Hz, detect_range=250 * ms,
                                                min_dist=sim_params['tau_l_pp'].get_param())
                                ],
                                plot_list=[]
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
