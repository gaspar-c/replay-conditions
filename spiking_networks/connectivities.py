"""
connectivities.py
Connectivity helpers used to build synaptic connections between populations.

This module defines a set of connectivity patterns:
- `ConnRandom`: uniform random connectivity between two populations.
- `ConnMorphSequence`: builds background, recurrent (within-assembly) and
    feedforward (between-assemblies) connections to create sequences of
    assemblies.
- `ConnRandomToSequence`: connects a homogeneous population to a population that
    contains assemblies (same connectivity for all postsynaptic assemblies).
"""

import time
import numpy as np
from brian2 import Synapses
from general_code.aux_functions import xprint


def gen_binary_array(array_size, prob):
    """
    Generate a boolean array where each entry has probability `prob` of being True.

    Parameters
    - array_size: int - number of elements in the returned boolean array
    - prob: float - probability in [0,1] for each element to be True

    Returns
    - numpy.ndarray of bool with shape `array_size`
    """

    integer_scale = 1000

    if int(prob * integer_scale) == 0:
        print('Error, probability precision is too small!')
        exit()

    # Use integer randoms for speed on very large arrays
    rand_array = np.random.randint(integer_scale, size=array_size, dtype=np.uint16)
    return rand_array < integer_scale * prob


class ConnectivityMatrix:
    """
    Builds connectivity matrix between two neuron populations based on a
    given connection probability. Supports fixed in-degree or variable-size
    (Bernoulli trials) connection schemes.

    The object stores the presynaptic and postsynaptic indices as 1D arrays
    (`pre_idx`, `post_idx`) and the total synapse count in `n_syn`.

    Parameters
    - p_ij: float - probability of connection from j->i
    - n_j: int - number of presynaptic neurons
    - n_i: int - number of postsynaptic neurons
    - name: str - label for this connection
    - fixed_size: bool - if True, enforce exactly expected in-degree per post
    """

    def __init__(self, p_ij, n_j, n_i, name, fixed_size=False):
        self.p_ij = p_ij
        self.n_j = int(n_j)
        self.n_i = int(n_i)
        self.name = name

        # generation metadata
        self.seed = None
        # arrays storing connection lists: pre_idx[k] -> post_idx[k]
        self.pre_idx = None
        self.post_idx = None
        self.n_syn = 0
        self.fixed_size = fixed_size

    def create_conn(self, seed, log=None):
        """
        Build the connection lists (presynaptic and postsynaptic indices).

        Two modes are supported:
        - fixed_size: each postsynaptic neuron receives exactly the expected
        number of incoming connections (in-degree = n_j * p_ij).
        - variable: a Bernoulli trial is performed for every possible pair
        (j,i). For large networks the function builds the connection list in
        chunks to avoid excessive memory usage.

        Parameters
        - seed: int - numpy seed used for reproducible generation
        - log: str or None - logfile path for progress messages
        """

        self.seed = seed
        np.random.seed(self.seed)

        # Fixed in-degree: sample presynaptic neurons with replacement for each
        # postsynaptic target so every post receives the same number of inputs.
        if self.fixed_size:
            n_syn_per_post = int(self.n_j * self.p_ij)
            self.n_syn = n_syn_per_post * self.n_i

            total_start_time = time.time()
            self.pre_idx = np.random.choice(np.arange(self.n_j), size=self.n_syn, replace=True)
            self.post_idx = np.repeat(np.arange(self.n_i), n_syn_per_post)
            xprint('\t Calculated %s synapses in %.0f seconds with fixed in-degree. '
                   'Each post-synaptic neuron received %s connections' %
                   ('{:,}'.format(self.n_syn), (time.time() - total_start_time), '{:,}'.format(n_syn_per_post)),
                   log)

        # Variable-size (Bernoulli trials for each potential connection)
        else:
            # total number of potential connections
            total_size = self.n_j * self.n_i

            # allocate a safety buffer (1.5x expected size) for indices
            synapses = np.zeros(int(total_size * self.p_ij * 1.5), dtype=np.uint64)

            # For very large networks we process the connection matrix in chunks
            # to limit memory use (max_size entries per chunk).
            max_size = int(2 * 1e9)

            total_start_time = time.time()
            count_step = 0
            count_syn = 0
            size_left = total_size
            if total_size > max_size:
                time_spent = 0

                xprint('\t calculating ca. %s out of %s (%s x %s) possible synapses...' %
                       ('{:,}'.format(int(total_size * self.p_ij)), '{:,}'.format(total_size),
                        '{:,}'.format(self.n_j), '{:,}'.format(self.n_i)),
                        log)
                while size_left > max_size:
                    start_time0 = time.time()

                    # generate a binary mask for this chunk and append non-zero indices
                    conn_matrix_part = gen_binary_array(max_size, self.p_ij)
                    non_zeros_part = count_step * max_size + np.nonzero(conn_matrix_part)[0]
                    synapses[count_syn:count_syn + len(non_zeros_part)] = non_zeros_part
                    count_syn += len(non_zeros_part)
                    count_step += 1

                    # update progress estimates
                    size_left = size_left - max_size
                    curr_percent = (1 - size_left / total_size) * 100
                    time_spent += (time.time() - start_time0)
                    time_left = (100 - curr_percent) * time_spent / curr_percent
                    xprint('\t\t %s: %.0f%% calculated in %.0f seconds (ca. %.0f seconds left...)' %
                           ('{:,}'.format(count_syn), curr_percent, time_spent, time_left), log)

            # final chunk (or only chunk when total_size <= max_size)
            conn_matrix_part = gen_binary_array(size_left, self.p_ij)
            non_zeros_part = count_step * max_size + np.nonzero(conn_matrix_part)[0]

            synapses[count_syn:count_syn + len(non_zeros_part)] = non_zeros_part
            count_syn += len(non_zeros_part)

            # trim allocation to actual number of discovered synapses
            synapses = synapses[:count_syn]

            xprint('\t %s: calculated all synapses in %.0f seconds.' %
                   ('{:,}'.format(count_syn), (time.time() - total_start_time)), log)

            # convert the flat indices into (pre, post) pairs
            self.pre_idx = np.array(synapses % self.n_j, dtype=int)
            self.post_idx = np.array(synapses // self.n_j, dtype=int)
            self.n_syn = len(self.pre_idx)


class ConnRandom:
    """Uniform random connectivity.

    Every possible pre->post pair is connected with probability `prob`. The
    implementation delegates the actual sampling to `ConnectivityMatrix` which
    returns explicit index lists used to call `Synapses.connect(i=..., j=...)`.

    The method supports an optional fixed-size connection mode 
    forwarded to `ConnectivityMatrix`.
    """
    def __init__(self,
                 name,
                 pre_name,
                 post_name,
                 conn_seed,
                 prob,
                 syn_model,
                 conn_fixed=False):
        """
        Args:
            name: name of connectivity object
            pre_name: name of presynaptic population
            post_name: name of postsynaptic population
            conn_seed: random seed
            prob: probability of connection
            syn_model: model for synaptic dynamics
            conn_fixed: [True/False], determines if the number of synapses is
                        always fixed (equal to the expected value)
        """
        self.name = name
        self.asb_flag = False
        self.pre_name = pre_name
        self.post_name = post_name
        self.conn_seed = conn_seed
        self.prob = prob
        self.syn_model = syn_model
        self.conn_fixed = conn_fixed

    def connect(self, built_network, log=None):
        """Create Synapses between two populations using uniform random wiring.

        Parameters
        - built_network: container mapping population names to Brian2 objects
        - log: optional logfile path used by `xprint`
        """

        str_pre = self.pre_name
        str_post = self.post_name
        str_ij = str_post + str_pre

        pop_pre = built_network['pop_' + str_pre]
        pop_post = built_network['pop_' + str_post]

        n_pre = pop_pre.N
        n_post = pop_post.N

        # ensure reproducible connectivity
        np.random.seed(self.conn_seed)

        xprint('%s->%s:' % (str_pre.upper(), str_post.upper()), log)
        if self.prob > 0:
            # build index lists for connections
            conn = ConnectivityMatrix(self.prob, n_pre, n_post, 'conn_' + str_ij,
                                      fixed_size=self.conn_fixed)
            conn.create_conn(self.conn_seed, log=log)

            # create Brian2 Synapses object and connect using index lists
            syn = Synapses(pop_pre, pop_post,
                           model=self.syn_model.model_eqs,
                           on_pre=self.syn_model.on_pre_eqs,
                           on_post=self.syn_model.on_post_eqs,
                           delay=self.syn_model.latency.get_param(),
                           method='euler',
                           name='syn_' + self.syn_model.name)
            syn.connect(i=conn.pre_idx, j=conn.post_idx)
            # apply synapse model parameters (weights, taus, etc.)
            self.syn_model.attr_params(syn)
            built_network.add(syn)

        else:
            xprint('\t no connections created; probability is 0', log)


class ConnMorphSequence:
    """Connectivity pattern that builds assemblies and feedforward links.

    This class constructs three connection types:
    - background (`prob_bg`) applied to all neurons
    - recurrent within-assembly (`prob_rc`) applied to each assembly subgroup
    - feedforward between successive assemblies (`prob_ff`)

    The method supports an optional fixed-size connection mode 
    forwarded to `ConnectivityMatrix`.
    """
    def __init__(self,
                 name,
                 pre_name,
                 post_name,
                 conn_seed,
                 prob_bg,
                 prob_rc,
                 prob_ff,
                 syn_model,
                 conn_fixed=False
                 ):
        """
        Args:
            name: name of connectivity object
            pre_name: name of presynaptic population
            post_name: name of postsynaptic population
            conn_seed: random seed
            prob_bg: background connection probability
            prob_rc: recurrent probability (within an assembly)
            prob_ff: feedforward probability (across assemblies)
            syn_model: model for synaptic dynamics
            conn_fixed: [True/False], determines if the number of synapses is
                        always fixed (equal to the expected value)
        """
        self.name = name
        self.asb_flag = True
        self.pre_name = pre_name
        self.post_name = post_name
        self.conn_seed = conn_seed
        self.prob_bg = prob_bg
        self.prob_rc = prob_rc
        self.prob_ff = prob_ff
        self.syn_model = syn_model
        self.conn_fixed = conn_fixed

    def connect(self, net_objects, built_network, log=None):

        # lookup population objects
        pop_pre = built_network['pop_' + self.pre_name]
        pop_post = built_network['pop_' + self.post_name]

        str_pre = self.pre_name
        str_post = self.post_name
        str_ij = str_post + str_pre

        n_pre = pop_pre.N
        n_post = pop_post.N

        # sizes of individual assemblies (number of neurons per assembly)
        n_pre_asb = net_objects.pop_settings[self.pre_name].asb_size
        n_post_asb = net_objects.pop_settings[self.post_name].asb_size

        # reproducible per-assembly seeds for ffw/rc connections
        np.random.seed(self.conn_seed)
        n_ffw = net_objects.n_asb - 1
        ff_seeds = np.random.randint(1, high=999, size=n_ffw, dtype=int)
        if self.prob_rc > 0:
            rc_seeds = np.random.randint(1, high=999, size=net_objects.n_asb, dtype=int)

        # background connections applied across the whole populations
        xprint('%s->%s (background):' % (str_pre.upper(), str_post.upper()), log)
        if self.prob_bg > 0:
            conn_bg = ConnectivityMatrix(self.prob_bg, n_pre, n_post, 'conn_' + str_ij + '_bg',
                                         fixed_size=self.conn_fixed)
            conn_bg.create_conn(self.conn_seed, log=log)

            syn_bg = Synapses(pop_pre, pop_post,
                              model=self.syn_model.model_eqs,
                              on_pre=self.syn_model.on_pre_eqs,
                              on_post=self.syn_model.on_post_eqs,
                              delay=self.syn_model.latency.get_param(),
                              method='euler',
                              name='syn_' + self.syn_model.name + '_bg')
            syn_bg.connect(i=conn_bg.pre_idx, j=conn_bg.post_idx)
            self.syn_model.attr_params(syn_bg)
            built_network.add(syn_bg)
        else:
            xprint('\t no connections created; probability is 0', log)

        # create per-assembly recurrent and feedforward connections
        for i in range(net_objects.n_asb):
            # recurrent (within-assembly) connections
            xprint('%s->%s (assembly %d):' % (str_pre.upper(), str_post.upper(), i + 1), log)

            if self.prob_rc > 0:
                # build connectivity for this assembly
                conn_rc_name = 'conn_' + str_ij + '_rc_' + str(i + 1)
                conn_rc = ConnectivityMatrix(self.prob_rc, n_pre_asb, n_post_asb, conn_rc_name,
                                             fixed_size=self.conn_fixed)
                conn_rc.create_conn(int(rc_seeds[i]), log=log)

                syn_rc = Synapses(pop_pre[i * n_pre_asb:(i + 1) * n_pre_asb],
                                  pop_post[i * n_post_asb:(i + 1) * n_post_asb],
                                  model=self.syn_model.model_eqs,
                                  on_pre=self.syn_model.on_pre_eqs,
                                  on_post=self.syn_model.on_post_eqs,
                                  delay=self.syn_model.latency.get_param(),
                                  method='euler',
                                  name='syn_' + self.syn_model.name + '_rc_' + str(i + 1))
                syn_rc.connect(i=conn_rc.pre_idx, j=conn_rc.post_idx)
                self.syn_model.attr_params(syn_rc)
                built_network.add(syn_rc)
            else:
                xprint('\t no recurrent connections created; probability is 0', log)

            # feedforward connections to the next assembly
            ff_pre_idx = i
            ff_post_idx = i + 1
            if i == net_objects.n_asb - 1:
                break

            xprint('%s (asb %d) -> %s (asb %d):' % (str_pre.upper(), ff_pre_idx + 1,
                                                    str_post.upper(), ff_post_idx + 1), log)

            if self.prob_ff > 0:
                conn_ff_name = 'conn_' + str_ij + '_ff_' + str(ff_pre_idx + 1)
                conn_ff = ConnectivityMatrix(self.prob_ff, n_pre_asb, n_post_asb, conn_ff_name,
                                             fixed_size=self.conn_fixed)
                conn_ff.create_conn(int(ff_seeds[i]), log=log)

                syn_ff = Synapses(pop_pre[ff_pre_idx * n_pre_asb:(ff_pre_idx + 1) * n_pre_asb],
                                  pop_post[ff_post_idx * n_post_asb:(ff_post_idx + 1) * n_post_asb],
                                  model=self.syn_model.model_eqs,
                                  on_pre=self.syn_model.on_pre_eqs,
                                  on_post=self.syn_model.on_post_eqs,
                                  delay=self.syn_model.latency.get_param(),
                                  method='euler',
                                  name='syn_' + self.syn_model.name + '_ff_' + str(ff_pre_idx + 1))
                syn_ff.connect(i=conn_ff.pre_idx, j=conn_ff.post_idx)
                self.syn_model.attr_params(syn_ff)
                built_network.add(syn_ff)
            else:
                xprint('\t no feedforward connections created; probability is 0', log)


class ConnRandomToSequence:
    """Connect a homogeneous population to an assembly-structured population.

    The same connectivity properties (`prob`) are applied for connections from
    the pre population to every postsynaptic assembly subgroup. Useful when a
    homogeneous population connects to a population with embedded assemblies.

    The method supports an optional fixed-size connection mode 
    forwarded to `ConnectivityMatrix`.
    """
    def __init__(self,
                 name,
                 pre_name,
                 post_name,
                 conn_seed,
                 prob,
                 syn_model,
                 conn_fixed=False
                 ):
        """
        Args:
            name: name of connectivity object
            pre_name: name of presynaptic population
            post_name: name of postsynaptic population
            conn_seed: random seed
            prob: connection probability
            syn_model: model for synaptic dynamics
            conn_fixed: [True/False], determines if the number of synapses is
                        always fixed (equal to the expected value)
        """
        self.name = name
        self.asb_flag = True
        self.pre_name = pre_name
        self.post_name = post_name
        self.conn_seed = conn_seed
        self.prob = prob
        self.syn_model = syn_model
        self.conn_fixed = conn_fixed

    def connect(self, net_objects, built_network, log=None):
        """Connect the pre population to each assembly subgroup of the post population.

        Parameters
        - net_objects: object containing population settings
        - built_network: mapping of population names to Brian2 population objects
        - log: optional logfile path
        """

        pop_pre = built_network['pop_' + self.pre_name]
        pop_post = built_network['pop_' + self.post_name]

        str_pre = self.pre_name
        str_post = self.post_name
        str_ij = str_post + str_pre

        n_pre = pop_pre.N
        n_post_asb = net_objects.pop_settings[self.post_name].asb_size

        # reproducible per-assembly seeds
        np.random.seed(self.conn_seed)
        rc_seeds = np.random.randint(1, high=999, size=net_objects.n_asb, dtype=int)

        # iterate over assemblies and create identical-style connections
        for i in range(net_objects.n_asb):
            xprint('%s->%s (assembly %d):' % (str_pre.upper(), str_post.upper(), i + 1), log)

            if self.prob > 0:
                conn_rc_name = 'conn_' + str_ij + '_rc_' + str(i + 1)
                conn_rc = ConnectivityMatrix(self.prob, n_pre, n_post_asb, conn_rc_name,
                                             fixed_size=self.conn_fixed)
                conn_rc.create_conn(int(rc_seeds[i]), log=log)

                syn_rc = Synapses(pop_pre,
                                  pop_post[i * n_post_asb:(i + 1) * n_post_asb],
                                  model=self.syn_model.model_eqs,
                                  on_pre=self.syn_model.on_pre_eqs,
                                  on_post=self.syn_model.on_post_eqs,
                                  delay=self.syn_model.latency.get_param(),
                                  method='euler',
                                  name='syn_' + self.syn_model.name + '_rc_' + str(i + 1))
                syn_rc.connect(i=conn_rc.pre_idx, j=conn_rc.post_idx)
                self.syn_model.attr_params(syn_rc)

                built_network.add(syn_rc)
            else:
                xprint('\t no recurrent connections created; probability is 0', log)
                