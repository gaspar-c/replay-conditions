from brian2 import *
import math
from my_code.simulations import initialize_sim
from my_code.aux_functions import xprint
from my_code.discrete_model import DiscreteSimulation, RectangleModel, StepDrive, test_replay, plot_bars
from my_code.parameters import Parameter


def run_simulation(options):

    sim_params = {
                  'check_theory': Parameter(True),
                  'w_rc': Parameter(1.0),
                  'w_ff': Parameter(1.0),
                  # 'j0': Parameter(1.1),
                  # 'j0_sigma': Parameter(0.0),
                  'j0': Parameter(1.0),
                  'j0_step': Parameter(1),
                  'v_width': Parameter(10, mV),
                  'v_decay': Parameter(0, mV),
                  'v_thres': Parameter(0.0, mV),
                  'n_asb': Parameter(10),
                  'max_speed_inv': Parameter(100),
                  'inf_model': Parameter(False),
    }

    log, sim_params, plot_params = initialize_sim(options, sim_params)
    n_asb = sim_params['n_asb'].get_param()
    inf_model = sim_params['inf_model'].get_param()

    # P connections
    v_width = sim_params['v_width'].get_param()
    w_rc = sim_params['w_rc'].get_param() * v_width
    w_ff = sim_params['w_ff'].get_param() * v_width
    v_decay = sim_params['v_decay'].get_param()
    v_thres = sim_params['v_thres'].get_param()
    vf_init = 0. * mV

    # current injected to first assembly at t = 0
    j0 = sim_params['j0'].get_param() * v_width

    print('w_rc = %f mV' % (w_rc / mV))
    print('w_ff = %f mV' % (w_ff / mV))

    t_stop = math.ceil(sim_params['max_speed_inv'].get_param() * n_asb)
    # t_stop = 60
    sim_0 = DiscreteSimulation(t_stop=t_stop,
                               model=RectangleModel(n_asb=n_asb,
                                                    w_rc=w_rc, w_ff=w_ff,
                                                    vf_init=vf_init,
                                                    v_thres=v_thres,
                                                    v_width=v_width,
                                                    v_decay=v_decay,
                                                    inf_model=inf_model),
                               ext_drive_asb1=StepDrive(start=1, stop=sim_params['j0_step'].get_param(),
                                                        amp=j0/sim_params['j0_step'].get_param()))
    sim_0.run_sim()

    test_replay(sim_0, options, sim_params, log=log, onlylast=False)

    if options['output_plots']:
        plot_bars(sim_0, options, plot_total=True, fold=False)

    xprint('\n=========== Used Simulation Parameters ============\n', log)
    for p in sim_params:
        if sim_params[p].used:
            xprint('%s = %s' % (p, sim_params[p].get_param()), log)

    xprint('\n========== Finished Printing Results ===========\n', log)
