from brian2 import *
import copy
from my_code.aux_functions import xprint
from my_code.simulations import initialize_sim, SimElement, run_network_sim
from my_code.parameters import Parameter
from my_code.tests import NetworkTests, TestReplay, TestFiring, VoltageStimulus, TestFitV
from my_code.network import VoltageStimulus
from my_code.plots import PlotRaster, PlotPopRate, PlotV, PlotV1D
from my_code import network as net, synapses as syn, connectivities as conn


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
                  'p_rc': Parameter(0.20),
                  'p_ff': Parameter(0.20),
                  'ff_loop': Parameter(False),
                  'rate_ext': Parameter(50, Hz),
                  'syn_weight_pe': Parameter(0.06, mV),
                  'g_pp': Parameter(0.10, nS),
                  'tau_l_pe': Parameter(1, ms),
                  'tau_l_pp': Parameter(1, ms),
                  'tau_ref': Parameter(1, ms),
                  'v_time': Parameter(16),
                  'conn_fixed': Parameter(False),
                  'g_leak': Parameter(10, nS),
                  'mem_cap': Parameter(200, pF),
                  'g_leak': Parameter(10, nS),
                  'curr_bg': Parameter(58, pA),
                  'stim_spread': Parameter(0, ms),
                  'stim_frac': Parameter(1.0),
    }

    log, sim_params, plot_params = initialize_sim(options, sim_params)
    network_objects = net.NetworkObjects()

    """ BUILD BRIAN NETWORK """
    xprint('================ Building Network ================', log)

    # external Poisson population
    pop_e_sett = net.PoissonPopulationSettings(name='e',
                                               n_cells=sim_params['n_e'].get_param(),
                                               rate=sim_params['rate_ext'].get_param(),
                                               plot_color='black')
    network_objects.add_to_pop(pop_e_sett)

    # create LIF model
    std_lif = net.LIFModel(mem_cap=sim_params['mem_cap'],
                           g_leak=sim_params['g_leak'],
                           e_rest=Parameter(-60, mV),
                           v_reset=Parameter(-60, mV),
                           v_thres=Parameter(-50, mV),
                           tau_refr=sim_params['tau_ref'],
                           # curr_bg=Parameter(58, pA)
                           curr_bg=sim_params['curr_bg']
                           )

    # population p
    pop_p_sett = net.NeuronPopulationSettings(name='p',
                                              model=copy.copy(std_lif),
                                              n_cells=sim_params['n_p'].get_param(),
                                              asb_flag=True,
                                              asb_size=sim_params['m_p'].get_param(),
                                              synapses=[
                                                  syn.PostSynapseCondSingExp(pre_name='p',
                                                                             e_nernst=Parameter(0, mV),
                                                                             tau_d=Parameter(2, ms)),
                                                  syn.PostSynapseNone(pre_name='e')],
                                              plot_color=plot_params['p_color'].get_param(),
                                              raster_height=4.0)
    network_objects.add_to_pop(pop_p_sett)

    # create brian network with defined populations
    brian_network = Network()
    pop_p_sett.create_brian_group(brian_network)
    pop_e_sett.create_brian_group(brian_network)

    # connectivity
    network_objects.n_asb = sim_params['n_asb'].get_param()

    syn_pp_sett = conn.ConnMorphSequence(name='pp',
                                         pre_name='p', post_name='p',
                                         conn_seed=sim_params['conn_seed'].get_param(),
                                         prob_bg=sim_params['p_bg'].get_param(),
                                         prob_rc=sim_params['p_rc'].get_param(),
                                         prob_ff=sim_params['p_ff'].get_param(),
                                         ffw_loop=sim_params['ff_loop'].get_param(),
                                         conn_fixed=sim_params['conn_fixed'].get_param(),
                                         syn_model=syn.SynapseCondNormal(
                                             name='pp',
                                             weight=sim_params['g_pp'],
                                             latency=sim_params['tau_l_pp'])
                                         )
    syn_pp_sett.connect(network_objects, brian_network, log=log)
    network_objects.add_to_syn(syn_pp_sett)

    syn_pe_sett = conn.ConnRandomToSequence(name='pe',
                                            pre_name='e', post_name='p',
                                            conn_seed=sim_params['conn_seed'].get_param(),
                                            prob=sim_params['p_pe'].get_param(),
                                            conn_fixed=False,
                                            # syn_model=syn.SynapseVoltageSTDP(
                                            #     name='pe',
                                            #     w_init=sim_params['syn_weight_pe'],
                                            #     latency=Parameter(1, ms),
                                            #     rho0=Parameter(2, Hz),
                                            #     tau=Parameter(20, ms))
                                            syn_model=syn.SynapseCurrentDirac(
                                                name='pe',
                                                weight=sim_params['syn_weight_pe'],
                                                latency=sim_params['tau_l_pe'])
                                           )
    syn_pe_sett.connect(network_objects, brian_network, log=log)
    network_objects.add_to_syn(syn_pe_sett)

    xprint('================ Network Built ================', log)

    """ SIMULATION EVENTS AND TESTS """

    events = []
    monitors = []

    wait_time = 1 * second

    # Test AI state
    # events.append(NetworkTests(monitors=monitors,
    #                            start=wait_time - 500 * ms, stop=wait_time,
    #                            max_record=sim_params['max_record'].get_param(),
    #                            test_list=[
    #                                TestFiring(pops=[pop_p_sett], store_group=True),
    #                                # TestBalance(pops=[pop_p_sett]),
    #                                TestFitV(pops=[pop_p_sett], asb=[10], time=[wait_time-150*ms]),
    #                                # TestSTDP(syns=['syn_pe_rc_10'], attr='weight_pe', unit=mV),
    #                           ],
    #                            plot_list=[
    #                               PlotRaster(pops=[pop_p_sett]),
    #                               PlotPopRate(pops=[pop_p_sett], filter_width=1*ms),
    #                               PlotV(pop=pop_p_sett, asb=[10]),
    #                               PlotV1D(pop=pop_p_sett, time=[wait_time-150*ms], asb=[10])
    #                           ],
    #                            n_time_ticks=4))


    # Test Replay
    inter_stim_time = 1 * second
    n_stims = sim_params['n_stims'].get_param()
    for stim_idx in range(n_stims):
        stim_time = wait_time + stim_idx * inter_stim_time

        # Stim assembly 1
        events.append(VoltageStimulus(target=pop_p_sett, time=stim_time, asb=1,
                                      frac=sim_params['stim_frac'].get_param(),
                                      spread=sim_params['stim_spread'].get_param())
                      )


        # Test if replay succeeds
        events.append(NetworkTests(monitors=monitors,
                                   start=stim_time - 50 * ms, stop=stim_time + 400 * ms,
                                   max_record=sim_params['max_record'].get_param(),
                                   test_list=[
                                      TestReplay(pop=pop_p_sett, filter_width=2 * ms,
                                                 detect_range=80 * ms, min_height=30*Hz,
                                                 min_dist=sim_params['tau_l_pp'].get_param())
                                   ],
                                   plot_list=[
                                   ],
                                   ))

        # Plot replay
        if options['output_plots']:
            plot_params['fig_height'] = Parameter(8, cm)
            plot_params['fig_width'] = Parameter(6, cm)
            plot_params['hide_time_ax'] = Parameter(True)
            events.append(NetworkTests(monitors=monitors,
                                       start=stim_time - 10 * ms, stop=stim_time + 70 * ms,
                                       max_record=sim_params['max_record'].get_param(),
                                       test_list=[
                                       ],
                                       plot_list=[
                                           PlotRaster(pops=[pop_p_sett], plot_bg=False, bg_in=False),
                                           PlotPopRate(pops=[pop_p_sett], filter_width=1*ms, y_max=[500],
                                                       title=r'\textbf{Assembly Rates (1/s)}'),
                                           PlotV(pop=pop_p_sett, asb=[10],
                                                 title=r'\textbf{Last Assembly $v$ (mV)}'),
                                           PlotV1D(pop=pop_p_sett,
                                                   time=[stim_time - 5*ms,
                                                         stim_time + sim_params['v_time'].get_param()*ms],
                                                   asb=[10])
                                       ],
                                       time_bar=20*ms
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

