"""
model_rectangle.py
Rectangle population model for replay simulations.

This module implements a population model where a distribution of membrane potentials
is represented by a clipped rectangle (uniform distribution). 
Assembly activation is modeled as the fraction of the distribution that crosses
the firing threshold, with dynamics governed by recurrent and feedforward weights.
"""

import math
from general_code.aux_functions import xprint
from population_model.simulations import DiscreteSimulation, StepDrive, test_replay, plot_bars
from general_code.parameters import Parameter, initialize_params


class RectangleModel:
    def __init__(self, n_asb,
                 w_rc=0, w_ff=0):
        """Initialize the rectangle population model.
        
        Args:
            n_asb (int): Number of assemblies in the sequence.
            w_rc (float): Recurrent connection weight (within-assembly).
            w_ff (float): Feedforward connection weight (between assemblies).
        """
        self.n_asb = n_asb
        self.w_rc = w_rc
        self.w_ff = w_ff

    def vf_next(self, v_pos, act_del, act_prev_del, drive_ext):
        """
        Update population activity position to the next time step.
        
        position advances via recurrent feedback from current assembly, 
        feedforward input from previous assembly, and external drive.
        
        Args:
            v_pos (float): Current position
            act_del (float): Delayed activity (from previous time step) from current (recurrent) assembly.
            act_prev_del (float): Delayed activity (from previous time step) from previous (feedforward) assembly.
            drive_ext (float): External drive input.
        
        Returns:
            float: Updated position, saturated at the threshold.
        """
        out = v_pos
        
        # Update position: recurrent + feedforward + external input
        out += self.w_rc * act_del + self.w_ff * act_prev_del + drive_ext

        # Saturate at firing threshold with small numerical margin (0.001% of width)
        if out >= 1.0:
            out = 1.000001

        return out

    def area_thres(self, vf_prev, vf_curr):
        """
        Calculate fraction of population crossing the activation threshold.
        
        For the rectangle model, activity is proportional to the distance traveled.
        The active fraction is simply the position difference (clamped to [0, 1]).
        
        Args:
            vf_prev (float): Position at previous time step.
            vf_curr (float): Position at current time step.
        
        Returns:
            float: Fraction of population that crossed threshold (0 to 1).
        """
        act = 0
        # Compute position change
        diff = (vf_curr - vf_prev)
        # Clamp to maximum possible activity (entire assembly width)
        if diff > 1:
            diff = 1
        # Only positive movement generates activity
        if diff > 0:
            act = diff
        return act
    

def run_simulation(options):
    """
    Execute a Gaussian population model simulation.
    
    Initializes simulation parameters, creates a discrete-time simulation with a
    Gaussian population model, runs the simulation, and analyzes replay propagation.
    Results are tested against theoretical predictions and optionally plotted.
    
    Args:
        options (dict): Dictionary containing simulation options, including:
            - 'output_plots': bool, whether to generate output plots.
    
    Returns:
        None. Results are logged and saved to disk.
    """
    # Define simulation parameters with default values
    sim_params = {
                  'w_rc': Parameter(1.0),           # Recurrent weight
                  'w_ff': Parameter(1.0),           # Feedforward weight
                  'n_asb': Parameter(10),           # Number of assemblies
                  'steps_per_asb': Parameter(100)   # Number of simulation time steps (per assembly)
    }

    # Initialize parameters from file/command-line options
    log, sim_params, plot_params = initialize_params(options, sim_params)
    n_asb = sim_params['n_asb'].get_param()

    # Extract connection weights (no scaling needed for rectangle model)
    w_rc = sim_params['w_rc'].get_param()
    w_ff = sim_params['w_ff'].get_param()

    print('w_rc = %f' % w_rc)
    print('w_ff = %f' % w_ff)

    # Calculate total simulation duration
    t_stop = math.ceil(sim_params['steps_per_asb'].get_param() * n_asb)

    # Create and configure the simulation with unit amplitude external drive
    sim_0 = DiscreteSimulation(t_stop=t_stop,
                               model=RectangleModel(n_asb=n_asb,
                                                    w_rc=w_rc, w_ff=w_ff),
                               ext_drive_asb1=StepDrive(start=1, 
                                                        stop=1,
                                                        amp=1))
    # Run the simulation
    sim_0.run_sim()

    # Analyze replay propagation and test against theory
    test_replay(sim_0, options, sim_params, log=log)

    # Generate visualization if requested (total activity, unfolded)
    if options['output_plots']:
        plot_bars(sim_0, options,  barcolor='#2d8b56cc')

    # Log final summary of parameters used
    xprint('\n=========== Used Simulation Parameters ============\n', log)
    for p in sim_params:
        if sim_params[p].used:
            xprint('%s = %s' % (p, sim_params[p].get_param()), log)

    xprint('\n========== Finished Printing Results ===========\n', log)
