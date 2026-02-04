"""
model_gauss.py
Gaussian population model for replay simulations.

This module implements a population model where a distribution of membrane potentials
is represented by a clipped Gaussian. Assembly activation is modeled as the
fraction of the distribution that crosses the firing threshold, 
with dynamics governed by recurrent and feedforward weights.
"""

import math as m
import numpy as np
from brian2 import mV
from general_code.aux_functions import xprint
from population_model.simulations import DiscreteSimulation, StepDrive, test_replay, plot_bars
from general_code.parameters import Parameter, initialize_params


class GaussModel:
    def __init__(self, n_asb,
                 w_rc=0., w_ff=0.,
                 n_std=4.):
        """
        Initialize the Gaussian population model.
        
        Args:
            n_asb (int): Number of assemblies.
            w_rc (float): Recurrent weight (strength of recurrent connections).
            w_ff (float): Feedforward weight (strength of feedforward connections).
            n_std (float): Width of the Gaussian distribution (in standard deviations).
        """
        self.n_asb = n_asb
        self.w_rc = w_rc
        self.w_ff = w_ff
        self.n_std = n_std

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

        # Update position: rightward movement driven by recurrent, feedforward, and external inputs
        out += self.w_rc * act_del + self.w_ff * act_prev_del + drive_ext

        # Saturate at firing threshold with small numerical margin (0.001% of width)
        if out >= self.n_std:
            out = self.n_std * 1.000001

        return out

    def area_thres(self, v_prev, v_curr):
        """
        Compute the fraction of population activity that crosses the threshold.
        
        Integrates the Gaussian distribution from the threshold (n_std/2) to infinity,
        computing the change in activated area as position moves from v_prev to v_curr.
        This represents the proportion of the distribution driven above firing threshold.
        
        Args:
            v_prev (float): Position at the previous time step.
            v_curr (float): Position at the current time step.
        
        Returns:
            float: Fraction of population activated by the threshold crossing (0 to 1).
        """
        act = 0
        
        # Compute total area under Gaussian (normalization factor)
        total_area = m.erf(self.n_std / (2 * np.sqrt(2)))
        
        # Compute area above threshold at previous position using error function
        area_prev = 1/2 * (1 - m.erf((self.n_std / 2 - v_prev) / (np.sqrt(2)))) / total_area
        
        # Compute area above threshold at current position
        area_curr = 1/2 * (1 - m.erf((self.n_std / 2 - v_curr) / (np.sqrt(2)))) / total_area
        
        # Activation is the change in area; clamp to [0, 1]
        diff = area_curr - area_prev
        if diff > 1:
            diff = 1
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
                  'n_std': Parameter(4.0),          # clipped-Gaussian width (in standard deviations)
                  'n_asb': Parameter(10),           # Number of assemblies
                  'steps_per_asb': Parameter(100),  # Number of simulation time steps (per assembly)
    }

    # Initialize parameters from file/command-line options
    log, sim_params, _ = initialize_params(options, sim_params)
    n_asb = sim_params['n_asb'].get_param()
    n_std = sim_params['n_std'].get_param()

    # Scale weights by clipped-Gaussian width for proper dynamics
    w_rc = sim_params['w_rc'].get_param() * n_std
    w_ff = sim_params['w_ff'].get_param() * n_std

    print('w_rc = %f' % w_rc)
    print('w_ff = %f' % w_ff)

    # Calculate total simulation duration
    t_stop = m.ceil(sim_params['steps_per_asb'].get_param() * n_asb)
    
    # Create and configure the simulation
    sim_0 = DiscreteSimulation(t_stop=t_stop,
                               model=GaussModel(n_asb=n_asb,
                                                w_rc=w_rc, w_ff=w_ff,
                                                n_std=n_std),
                               ext_drive_asb1=StepDrive(start=1, stop=1, amp=n_std))
    
    # Run the simulation
    sim_0.run_sim()

    # Analyze replay propagation and test against theory
    test_replay(sim_0, options, sim_params, log=log)

    # Generate visualization if requested
    if options['output_plots']:
        plot_bars(sim_0, options, barcolor='#2176b3ff')

    # Log final summary of parameters used
    xprint('\n=========== Used Simulation Parameters ============\n', log)
    for p in sim_params:
        if sim_params[p].used:
            xprint('%s = %s' % (p, sim_params[p].get_param()), log)

    xprint('\n========== Finished Printing Results ===========\n', log)
