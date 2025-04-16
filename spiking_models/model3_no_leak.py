from brian2 import *
import copy
from my_code.aux_functions import xprint
from my_code.simulations import initialize_sim, SimElement, run_network_sim
from my_code.parameters import Parameter
from my_code.tests import NetworkTests, TestBalance, TestReplay, TestFiring, TestFitV, TestSTDP
from my_code import network as net, synapses as syn, connectivities as conn
from my_code.network import VoltageStimulus



def run_simulation(options):

    sim_params = {
                  'n_stims': Parameter(5),
                  'conn_seed': Parameter(1),
                  'sim_seed': Parameter(1),
                  'init_state': Parameter('normal'),
                  'init_normal_sigma': Parameter(0.5, mV),
                  'init_normal_mean': Parameter(-51.9, mV),
                  'n_p': Parameter(5000),
                  'm_p': Parameter(500),
                  'n_asb': Parameter(10),
                  'p_bg': Parameter(0.00),
                  'p_rc': Parameter(0.00),
                  'p_ff': Parameter(0.12),
                  'g_pp': Parameter(0.10, nS),
                  'tau_l_pp': Parameter(1, ms)
    }

    log, sim_params, plot_params = initialize_sim(options, sim_params)
    network_objects = net.NetworkObjects()

    """ BUILD BRIAN NETWORK """
    xprint('================ Building Network ================', log)

    # create LIF model
    std_lif = net.LIFModel(mem_cap=Parameter(200, pfarad),
                           g_leak=Parameter(0, nS),
                           e_rest=Parameter(-60, mV),
                           v_reset=Parameter(-60, mV),
                           v_thres=Parameter(-50, mV),
                           tau_refr=Parameter(1, ms),
                           curr_bg=Parameter(0, pA))

    # population p
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

    # create brian network with defined populations
    brian_network = Network()
    pop_p_sett.create_brian_group(brian_network)

    # connectivity
    network_objects.n_asb = sim_params['n_asb'].get_param()

    syn_pp_sett = conn.ConnMorphSequence(name='pp',
                                         pre_name='p', post_name='p',
                                         conn_seed=sim_params['conn_seed'].get_param(),
                                         prob_bg=sim_params['p_bg'].get_param(),
                                         prob_rc=sim_params['p_rc'].get_param(),
                                         prob_ff=sim_params['p_ff'].get_param(),
                                         ffw_loop=False,
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

    wait_time = 1 * second
    # events.append(NetworkTests(monitors=monitors,
    #                            start=wait_time - 100 * ms, stop=wait_time,
    #                            max_record=sim_params['max_record'].get_param(),
    #                            test_list=[
    #                                TestFiring(pops=[pop_p_sett]),
    #                                TestBalance(pops=[pop_p_sett]),
    #                                # TestFitV(pops=[pop_p_sett], asb=[10], time=[wait_time-150*ms]),
    #                                # TestSTDP(syns=['syn_pe_rc_10'], attr='weight_pe', unit=mV),
    #                           ],
    #                            plot_list=[
    #                               # PlotRaster(pops=[pop_p_sett]),
    #                               # PlotPopRate(pops=[pop_p_sett], filter_width=1*ms),
    #                               # PlotV(pop=pop_p_sett, asb=[10]),
    #                               # PlotV1D(pop=pop_p_sett, time=[wait_time-150*ms], asb=[10])
    #                           ],
    #                            n_time_ticks=3))


    # Test Replay
    inter_stim_time = 1 * second
    n_stims = sim_params['n_stims'].get_param()
    for stim_idx in range(n_stims):
        stim_time = wait_time + stim_idx * inter_stim_time

        # Stim assembly 1
        events.append(VoltageStimulus(target=pop_p_sett, time=stim_time, asb=1, frac=1.0))

        # Test if replay succeeds
        events.append(NetworkTests(monitors=monitors,
                                   start=stim_time - 0.05 * second, stop=stim_time + 1 * second,
                                   max_record=sim_params['max_record'].get_param(),
                                   test_list=[
                                      TestReplay(pop=pop_p_sett, filter_width=7 * ms,
                                                 min_height=10*Hz, detect_range=250 * ms,
                                                 min_dist=sim_params['tau_l_pp'].get_param())
                                   ],
                                   plot_list=[
                                   ]
                                   ))

    # run network simulation:
    sim_settings = SimElement(options=options,
                              brian_net=brian_network,
                              net_objects=network_objects,
                              sim_params=sim_params,
                              plot_params=plot_params,
                              events=events,
                              monitors=monitors,
                              log=log)
    run_network_sim(sim_settings)
