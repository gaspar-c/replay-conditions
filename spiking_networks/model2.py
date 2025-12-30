"""
model2.py
Spiking neural network model with principal and basket cell populations, using Brian2.
Assemblies only contain principal cells; basket cells connect to all assemblies uniformly.
"""
import copy
from brian2 import Network, Hz, mV, nS, ms, second, pF, pA
from general_code.aux_functions import xprint
from spiking_networks.simulations import SimElement, run_network_sim
from general_code.parameters import Parameter, initialize_params
from spiking_networks.tests import NetworkTests, TestReplay, TriggerSpikes, TestFitV
from spiking_networks.network import ChangeAttribute
from spiking_networks.plot_spiking_trace import PlotRaster, PlotPopRate, PlotV, PlotV1D
from spiking_networks import network as net, synapses as syn, connectivities as conn



def run_simulation(options):

    sim_params = {
        'n_stims': Parameter(1),
        'stdp_time': Parameter(5, second),
        'test_ai': Parameter(False),
        'n_p': Parameter(20000),
        'm_p': Parameter(500),
        'n_b': Parameter(5000),
        'conn_seed': Parameter(2),
        'n_asb': Parameter(10),
        'p_bg': Parameter(0.01),
        'p_rc': Parameter(0.08),
        'p_ff': Parameter(0.08),
        'p_bp': Parameter(0.01),
        'p_bb': Parameter(0.01),
        'p_pb': Parameter(0.01),
        'g_pp': Parameter(0.10, nS),
        'g_bp': Parameter(0.10, nS),
        'g_pb_init': Parameter(0.40, nS),
        'g_bb': Parameter(0.40, nS),
        'tau_l': Parameter(1, ms),
        'tau_l_pp': Parameter(1, ms),
        'v_time': Parameter(19),
        'conn_fixed': Parameter(False),
    }

    log, sim_params, plot_params = initialize_params(options, sim_params)
    network_objects = net.NetworkObjects()

    """ BUILD BRIAN NETWORK """
    xprint('================ Building Network ================', log)

    # Create standard LIF neuron model
    std_lif = net.LIFModel(mem_cap=Parameter(200, pF),
                           g_leak=Parameter(10, nS),
                           e_rest=Parameter(-60, mV),
                           v_reset=Parameter(-60, mV),
                           v_thres=Parameter(-50, mV),
                           tau_refr=Parameter(1, ms),
                           curr_bg=Parameter(200, pA))

    # Define population 'p' (principal cells)
    pop_p_sett = net.NeuronPopulationSettings(name='p',
                                              model=copy.copy(std_lif),
                                              n_cells=sim_params['n_p'].get_param(),
                                              asb_flag=True,
                                              asb_size=sim_params['m_p'].get_param(),
                                              synapses=[
                                                syn.PostSynapseCondSingExp(pre_name='p',
                                                                           e_nernst=Parameter(0, mV),
                                                                           tau_d=Parameter(2, ms)),
                                                syn.PostSynapseCondSingExp(pre_name='b',
                                                                           e_nernst=Parameter(-80, mV),
                                                                           tau_d=Parameter(4, ms))],
                                              plot_color=plot_params['p_color'].get_param(),
                                              raster_height=4.0,
                                              n_raster=500)
    network_objects.add_to_pop(pop_p_sett)

    # Define population 'b' (basket cells)
    b_lif = copy.copy(std_lif)
    pop_b_sett = net.NeuronPopulationSettings(name='b',
                                              model=b_lif,
                                              n_cells=sim_params['n_b'].get_param(),
                                              synapses=[
                                                syn.PostSynapseCondSingExp(pre_name='p',
                                                                           e_nernst=Parameter(0, mV),
                                                                           tau_d=Parameter(2, ms)),
                                                syn.PostSynapseCondSingExp(pre_name='b',
                                                                           e_nernst=Parameter(-80, mV),
                                                                           tau_d=Parameter(4, ms))],
                                              plot_color=plot_params['b_color'].get_param())
    network_objects.add_to_pop(pop_b_sett)

    # Create Brian2 network and add populations
    brian_network = Network()
    pop_p_sett.create_brian_group(brian_network)
    pop_b_sett.create_brian_group(brian_network)

    # Define and connect synaptic pathways
    network_objects.n_asb = sim_params['n_asb'].get_param()

    syn_pp_sett = conn.ConnMorphSequence(name='pp',
                                         pre_name='p', post_name='p',
                                         conn_seed=sim_params['conn_seed'].get_param(),
                                         prob_bg=sim_params['p_bg'].get_param(),
                                         prob_rc=sim_params['p_rc'].get_param(),
                                         prob_ff=sim_params['p_ff'].get_param(),
                                         conn_fixed=sim_params['conn_fixed'].get_param(),
                                         syn_model=syn.SynapseCondNormal(
                                             name='pp',
                                             weight=sim_params['g_pp'],
                                             latency=sim_params['tau_l_pp']))
    syn_pp_sett.connect(network_objects, brian_network, log=log)
    network_objects.add_to_syn(syn_pp_sett)

    syn_bp_sett = conn.ConnRandom(name='bp',
                                  pre_name='p', post_name='b',
                                  conn_seed=sim_params['conn_seed'].get_param(),
                                  prob=sim_params['p_bp'].get_param(),
                                  conn_fixed=sim_params['conn_fixed'].get_param(),
                                  syn_model=syn.SynapseCondNormal(
                                      name='bp',
                                      weight=sim_params['g_bp'],
                                      latency=sim_params['tau_l']))
    syn_bp_sett.connect(brian_network, log=log)
    network_objects.add_to_syn(syn_bp_sett)

    syn_pb_sett = conn.ConnRandom(name='pb',
                                  pre_name='b', post_name='p',
                                  conn_seed=sim_params['conn_seed'].get_param(),
                                  prob=sim_params['p_pb'].get_param(),
                                  conn_fixed=sim_params['conn_fixed'].get_param(),
                                  syn_model=syn.SynapseCondSTDP(
                                      name='pb',
                                      g_init=sim_params['g_pb_init'],
                                      latency=sim_params['tau_l'],
                                      rho0=Parameter(1, Hz),
                                      tau=Parameter(20, ms)))
    syn_pb_sett.connect(brian_network, log=log)
    network_objects.add_to_syn(syn_pb_sett)

    syn_bb_sett = conn.ConnRandom(name='bb',
                                  pre_name='b', post_name='b',
                                  conn_seed=sim_params['conn_seed'].get_param(),
                                  prob=sim_params['p_bb'].get_param(),
                                  conn_fixed=sim_params['conn_fixed'].get_param(),
                                  syn_model=syn.SynapseCondNormal(
                                      name='bb',
                                      weight=sim_params['g_bb'],
                                      latency=sim_params['tau_l']))
    syn_bb_sett.connect(brian_network, log=log)
    network_objects.add_to_syn(syn_bb_sett)

    xprint('================ Network Built ================', log)

    """ SIMULATION EVENTS AND TESTS """

    events = []
    monitors = []
    stdp_on_time = 0 * second
    stdp_off_time = sim_params['stdp_time'].get_param()

    syn_pb_pops = ['syn_pb']
    for target_pop in syn_pb_pops:
        events.append(ChangeAttribute(onset=stdp_on_time,
                                      target=target_pop,
                                      attribute='eta',
                                      value=0.01))

        events.append(ChangeAttribute(onset=stdp_off_time,
                                      target=target_pop,
                                      attribute='eta',
                                      value=0.00))

    

    # Test AI state
    test_ai = sim_params['test_ai'].get_param()
    if test_ai:
        events.append(NetworkTests(monitors=monitors,
                                start=stdp_off_time - 0.5 * second, stop=stdp_off_time,
                                max_record=sim_params['max_record'].get_param(),
                                test_list=[
                                    TestFitV(pops=[pop_p_sett], asb=[10], time=[stdp_off_time]),
                                ],
                                plot_list=[]
                                ))
    
    # Test Replay
    wait_time = 1 * second
    n_stims = sim_params['n_stims'].get_param()
    for idx in range(n_stims):
        stim_time = stdp_off_time + idx * wait_time + 0.1 * second

        events.append(TriggerSpikes(target=pop_p_sett, time=stim_time, asb=1))

        # Test if replay succeeds
        events.append(NetworkTests(monitors=monitors,
                                   start=stim_time - 50 * ms, stop=stim_time + 400 * ms,
                                   max_record=sim_params['max_record'].get_param(),
                                   test_list=[
                                       TestReplay(pop=pop_p_sett, filter_width=2*ms, min_height=30*Hz,
                                                  detect_range=80*ms, min_dist=sim_params['tau_l_pp'].get_param())
                                   ],
                                   plot_list=[],
                                   ))

        # Plot replay
        if options['output_plots']:
            events.append(NetworkTests(monitors=monitors,
                                       start=stim_time - 10 * ms, stop=stim_time + 70 * ms,
                                       max_record=sim_params['max_record'].get_param(),
                                       test_list=[
                                       ],
                                       plot_list=[
                                           PlotRaster(pops=[pop_p_sett]),
                                           PlotPopRate(pops=[pop_p_sett]),
                                           PlotV(pop=pop_p_sett, asb=[10],
                                                 title=r'Last Assembly $v$ (mV)'),
                                           PlotV1D(pop=pop_p_sett,
                                                   time=[stim_time - 5 * ms,
                                                         stim_time + sim_params['v_time'].get_param() * ms],
                                                   asb=[10])
                                       ],
                                       time_bar=20 * ms
                                       ))

    sim_settings = SimElement(options=options,
                              brian_net=brian_network,
                              net_objects=network_objects,
                              sim_params=sim_params,
                              plot_params=plot_params,
                              events=events,
                              monitors=monitors,
                              log=log)

    # Run network simulation
    run_network_sim(sim_settings)
