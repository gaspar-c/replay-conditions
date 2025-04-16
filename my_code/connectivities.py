from brian2 import *
import numpy as np
from my_code.aux_functions import ConnectivityMatrix, xprint


class ConnRandom:
    """
    random morphological connectivity,
    all neurons have a fixed probability of connecting to all other neurons
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
        """
        self.name = name
        self.asb_flag = False
        self.pre_name = pre_name
        self.post_name = post_name
        self.conn_seed = conn_seed
        self.prob = prob
        self.syn_model = syn_model
        self.conn_fixed = conn_fixed

    def connect(self, built_network, conn_to_file=False, log=None):
        str_pre = self.pre_name
        str_post = self.post_name
        str_ij = str_post + str_pre

        pop_pre = built_network['pop_' + str_pre]
        pop_post = built_network['pop_' + str_post]

        n_pre = pop_pre.N
        n_post = pop_post.N

        np.random.seed(self.conn_seed)

        # if there are no assemblies by imposed connectivity:
        xprint('%s->%s:' % (str_pre.upper(), str_post.upper()), log)
        if self.prob > 0:
            conn = ConnectivityMatrix(self.prob, n_pre, n_post, 'conn_' + str_ij,
                                      fixed_size=self.conn_fixed, to_file=conn_to_file)
            conn.create_conn(self.conn_seed, log=log)

            syn = Synapses(pop_pre, pop_post,
                           model=self.syn_model.model_eqs,
                           on_pre=self.syn_model.on_pre_eqs,
                           on_post=self.syn_model.on_post_eqs,
                           delay=self.syn_model.latency.get_param(),
                           method='euler',
                           name='syn_' + self.syn_model.name)
            syn.connect(i=conn.pre_idx, j=conn.post_idx)
            self.syn_model.attr_params(syn)
            built_network.add(syn)

        else:
            xprint('\t no connections created; probability is 0', log)


class ConnMorphSequence:
    """
    creates a sequence of assemblies within a network by connecting
    subgroups of neurons with different probabilities
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
                 ffw_loop=False,
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
            ffw_loop: [True/False], determines if the last assembly
                      connects to the first one, forming a loop
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
        self.ffw_loop = ffw_loop
        self.conn_fixed = conn_fixed

    def connect(self, net_objects, built_network,
                asb_sleep=False, conn_to_file=False, log=None):

        pop_pre = built_network['pop_' + self.pre_name]
        pop_post = built_network['pop_' + self.post_name]

        str_pre = self.pre_name
        str_post = self.post_name
        str_ij = str_post + str_pre

        n_pre = pop_pre.N
        n_post = pop_post.N

        n_pre_asb = net_objects.pop_settings[self.pre_name].asb_size
        n_post_asb = net_objects.pop_settings[self.post_name].asb_size

        np.random.seed(self.conn_seed)

        if self.ffw_loop:
            n_ffw = net_objects.n_asb
        else:
            n_ffw = net_objects.n_asb - 1

        ff_seeds = np.random.randint(1, high=999, size=n_ffw, dtype=int)
        if self.prob_rc > 0:
            rc_seeds = np.random.randint(1, high=999, size=net_objects.n_asb, dtype=int)

        # background connections
        xprint('%s->%s (background):' % (str_pre.upper(), str_post.upper()), log)
        if self.prob_bg > 0:
            conn_bg = ConnectivityMatrix(self.prob_bg, n_pre, n_post, 'conn_' + str_ij + '_bg',
                                         fixed_size=self.conn_fixed, to_file=conn_to_file)
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

        # create assemblies:
        for i in range(net_objects.n_asb):
            # recurrent connections:
            xprint('%s->%s (assembly %d):' % (str_pre.upper(), str_post.upper(), i + 1), log)

            if self.prob_rc > 0:
                # impose specified connectivity:
                conn_rc_name = 'conn_' + str_ij + '_rc_' + str(i + 1)
                conn_rc = ConnectivityMatrix(self.prob_rc, n_pre_asb, n_post_asb, conn_rc_name,
                                             fixed_size=self.conn_fixed, to_file=conn_to_file)
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

                if asb_sleep:
                    setattr(syn_rc, 'g_%s' % str_ij, 0 * siemens)

                built_network.add(syn_rc)
            else:
                xprint('\t no recurrent connections created; probability is 0', log)

            # feedforward connections:
            ff_pre_idx = i
            ff_post_idx = i + 1
            if i == net_objects.n_asb - 1:
                if self.ffw_loop:
                    ff_post_idx = 0
                else:
                    break

            xprint('%s (asb %d) -> %s (asb %d):' % (str_pre.upper(), ff_pre_idx + 1,
                                                    str_post.upper(), ff_post_idx + 1), log)

            if self.prob_ff > 0:
                # imposed specified connectivity:
                conn_ff_name = 'conn_' + str_ij + '_ff_' + str(ff_pre_idx + 1)
                conn_ff = ConnectivityMatrix(self.prob_ff, n_pre_asb, n_post_asb, conn_ff_name,
                                             fixed_size=self.conn_fixed, to_file=conn_to_file)
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

                if asb_sleep:
                    setattr(syn_ff, 'g_%s' % str_ij, 0 * siemens)
                built_network.add(syn_ff)
            else:
                xprint('\t no feedforward connections created; probability is 0', log)


class ConnRandomToSequence:
    """
    connects a population without assemblies to a population with assemblies.
    Connections have the same properties for all postsynaptic assemblies
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
        """
        self.name = name
        self.asb_flag = True
        self.pre_name = pre_name
        self.post_name = post_name
        self.conn_seed = conn_seed
        self.prob = prob
        self.syn_model = syn_model
        self.conn_fixed = conn_fixed

    def connect(self, net_objects, built_network,
                asb_sleep=False, conn_to_file=False, log=None):

        pop_pre = built_network['pop_' + self.pre_name]
        pop_post = built_network['pop_' + self.post_name]

        str_pre = self.pre_name
        str_post = self.post_name
        str_ij = str_post + str_pre

        n_pre = pop_pre.N
        n_post_asb = net_objects.pop_settings[self.post_name].asb_size

        np.random.seed(self.conn_seed)

        rc_seeds = np.random.randint(1, high=999, size=net_objects.n_asb, dtype=int)

        # create assemblies:
        for i in range(net_objects.n_asb):
            # recurrent connections:
            xprint('%s->%s (assembly %d):' % (str_pre.upper(), str_post.upper(), i + 1), log)

            if self.prob > 0:
                # impose specified connectivity:
                conn_rc_name = 'conn_' + str_ij + '_rc_' + str(i + 1)
                conn_rc = ConnectivityMatrix(self.prob, n_pre, n_post_asb, conn_rc_name,
                                             fixed_size=self.conn_fixed, to_file=conn_to_file)
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

                if asb_sleep:
                    setattr(syn_rc, 'g_%s' % str_ij, 0 * siemens)

                built_network.add(syn_rc)
            else:
                xprint('\t no recurrent connections created; probability is 0', log)