from app.models import Regular, Synapse, SNPSystem
from app.utils import rule_dict_lookup, check_rule_validity, parse_rule
import numpy as np
import random
from typing import Dict, Union
import re


class MatrixSNPSystem:
    def __init__(self, system: SNPSystem):
        self.neurons = system.neurons
        self.synapses = system.synapses
        self.expected = system.expected
        self.rule_dict = system.rule_dict

        self.__set_neuron_order()
        self.__rule_dict_parse()
        self.__set_rule_order()

        self.config_vct = self.__init_config_vct()
        self.spike_train_vct = self.__init_spike_train_vct()
        self.adj_mx = self.__init_adj_mx()
        self.trans_mx = self.__init_trans_mx()
        self.delay_status_vct = self.__init_delay_status_vct()
        self.neuron_status_vct = self.__init_neuron_status_vct()
        self.delayed_indicator_vct = self.__init_delayed_indicator_vct()
        self.config_label_vct = self.__init_config_label_vct()
        self.mask_vct = self.__init_mask_vct()
        self.checker_vct = self.__init_checker_vct()
        self.graphs = self.__init_graphs()

        self.neuron_labels = [self.neuron_keys] 
        self.contents = [['' for i in range(self.neuron_count)]]
        self.states = [[0 for i in range(self.neuron_count)]]
        self.delays = [[0 for i in range(self.neuron_count)]]
        self.decisions = [[None for i in range(self.neuron_count)]]
        self.delay = [0 for i in range(self.neuron_count)]
        self.halted = False
        self.iteration = 0
        self.cursor = 0
        self.offset = 0

        # self.print_system()
        # print("++")

    def __repr__(self):
        return (
            f"ITERATION {self.iteration}\n"
            f"\n"
            f"Neuron Count:\n"
            f"{self.neuron_count}\n"
            f"Rule Count:\n"
            f"{self.rule_count}\n"
            f"Neurons:\n"
            f"{self.neurons}\n"
            f"Rules:\n"
            f"{self.rules}\n"
            f"Synapses:\n"
            f"{self.synapses}\n"
            f"Neuron Keys:\n"
            f"{self.neuron_keys}\n"
            f"Rule Keys:\n"
            f"{self.rule_keys}\n"
            f"Initial Configuration:\n"
            f"{self.config_vct}\n"
            f"Neuron Rule Map:\n"
            f"{self.neuron_rule_map}\n"
            f"Adjacency Matrix:\n"
            f"{self.adj_mx}\n"
            f"Spiking Transition Matrix:\n"
            f"{self.trans_mx}\n"
            f"Rule Delays:\n"
            f"{self.rule_delay_vct}\n"
            f"Config Label Vector:\n"
            f"{self.config_label_vct}\n"
            f"Mask Vector:\n"
            f"{self.mask_vct}\n"
            f"Spike Train Vector:\n"
            f"{self.spike_train_vct}\n"
            f"Spikeable Matrix:\n"
            f"{self.spikeable_mx}\n"
            f"Decision Vector:\n"
            f"{self.decision_vct}\n"
            f"Checker Vector:\n"
            f"{self.checker_vct}\n"
        )

    def pseudorandom_simulate_next(self):
        self.compute_spikeable_mx()
        self.__choose_decision_vct()
        self.compute_next_configuration()

    def compute_next_configuration(self):
        self.cursor += 1
        if self.iteration == self.cursor - 1:
            self.iteration += 1
            # Perform neuron division if there are any applicable
            self.__update_config_label_vct()
            self.__update_mask_vct()
            self.__update_checker_vct()
            self.__reset_offset()
            self.__division_algorithm()
            # Perform the rest of the simulation
            self.__update_delay_status_vct()
            self.__compute_indicator_vct()
            self.__check_halt_conditions()
            self.__update_delayed_indicator_vct()
            self.__update_neuron_status_vct()
            self.__add_input_spiketrain()
            self.__update_config_vct()
            self.__update_output_spiketrain()
            self.__update_neuron_states()
            self.__update_content()
            self.__update_decisions()
            self.__update_neuron_labels()
            self.__update_graphs()
        else:
            self.content = self.contents[self.cursor - 1]
            self.delay = self.delays[self.cursor - 1]
            self.state = self.states[self.cursor - 1]
            self.graph = self.graphs[self.cursor - 1]

    def compute_prev_configuration(self):
        if self.cursor == 1:
            return
        self.cursor -= 1
        self.content = self.contents[self.cursor - 1]
        self.delay = self.delays[self.cursor - 1]
        self.state = self.states[self.cursor - 1]
        self.graph = self.graphs[self.cursor - 1]

    def pseudorandom_simulate_all(self):
        iteration = 0
        while True:
            self.pseudorandom_simulate_next()

            if self.halted:
                break

            iteration += 1

    def validate_result(self):
        """
        Validates the result of the simulation
        """
        for i in range(self.neuron_count):
            if re.match(str(self.expected[i]), str(self.content[i])) is None:
                return False

        return True

    def __update_decisions(self):
        decision = [None] * self.neuron_count
        for neuron in self.neuron_rule_map:
            if neuron in self.input_keys:
                decision[neuron] = self.content[neuron]
                continue
            if neuron in self.output_keys:
                decision[neuron] = self.content[neuron]
                continue

            for rule in self.neuron_rule_map[neuron]:
                if self.decision_vct[rule]:
                    decision[neuron] = self.neuron_rule_map[neuron].index(rule)
                    break

        self.decisions.append(decision)

    def __check_halt_conditions(self):
        self.halted = (
            np.all(self.indicator_vct == 0)
            and np.all(self.delay_status_vct == 0)
            and np.all(self.spike_train_vct[self.input_keys] == "")
        ) or self.iteration == 1024

    def __update_content(self):
        self.content = self.config_vct.tolist()
        for i in self.input_keys: self.content[i] = self.spike_train_vct[i]
        for i in self.output_keys: self.content[i] = self.spike_train_vct[i]

        self.contents.append(self.content)

    def __add_input_spiketrain(self):
        self.spike_train = np.zeros(self.neuron_count, dtype=object)
        non_empty = np.where(self.spike_train_vct != "")[0]
        non_empty_inputs = np.intersect1d(non_empty, self.input_keys)

        for i in non_empty_inputs:
            spike = int(self.spike_train_vct[i][0] or 0)
            self.spike_train += self.adj_mx[i] * spike
            self.spike_train_vct[i] = str(self.spike_train_vct[i])[1:]

    def __update_output_spiketrain(self):
        spike = np.char.add(
            self.spike_train_vct[self.output_keys].astype(str),
            self.config_vct[self.output_keys].astype(str),
        )
        self.spike_train_vct[self.output_keys] = spike
        self.config_vct[self.output_keys] = 0

    def __update_neuron_states(self):
        self.state = [-1 if i == 0 else 0 for i in self.neuron_status_vct]

        for rule_idx in range(len(self.indicator_vct)):
            if self.indicator_vct[rule_idx] == 1:
                neuron_idx = self.__get_mapped_neuron(rule_idx)
                self.state[neuron_idx] = (
                    1 if self.rules[f"r{rule_idx}"]["type"] == "spiking" else 2
                )
        for i in self.input_keys:
            if self.spike_train_vct[i] == "":
                continue
            spike = str(self.spike_train_vct[i])[0]
            spike = int(spike) if spike else 0
            self.state[i] = spike

        self.states.append(self.state)

    def __update_config_vct(self):
        net_gain = (
            np.dot(self.indicator_vct, self.trans_mx).astype(int) + self.spike_train
        )
        self.config_vct += np.multiply(self.neuron_status_vct, net_gain).astype(int)

    def __choose_decision_vct(self):
        possible_spiking_vct = len(self.spikeable_mx)
        decision_vct_idx = random.randint(1, possible_spiking_vct) - 1

        self.decision_vct = self.spikeable_mx[decision_vct_idx]

    def __update_delayed_indicator_vct(self):
        self.delayed_indicator_vct = np.logical_or(
            np.logical_and(
                self.delayed_indicator_vct, np.logical_not(self.indicator_vct)
            ),
            np.logical_and(self.decision_vct, self.delay_status_vct > 0),
        ).astype(int)

    def __compute_indicator_vct(self):
        """
        Computes the indicator vector by multiplying
        the rule status vector with the chosen
        spiking vector
        """

        rule_status = np.logical_not(self.delay_status_vct).astype(int)

        self.indicator_vct = np.multiply(
            self.decision_vct + self.delayed_indicator_vct, rule_status
        )

    def __update_neuron_status_vct(self):
        """
        Updates the neuron status vector by setting the value to 0
        if the corresponding rule is not activatable
        """
        self.neuron_status_vct = np.ones((self.neuron_count,)).astype(int)

        for neuron_idx in range(self.neuron_count):
            if self.neuron_rule_map[neuron_idx] != []:
                rule_ref = self.neuron_rule_map[neuron_idx][0]
                delay = self.delay_status_vct[rule_ref]
                self.delay[neuron_idx] = delay
                self.neuron_status_vct[neuron_idx] = delay == 0
        self.delays.append(self.delay)

    def __update_delay_status_vct(self):
        """
        Updates the delay status vector by decrementing each value by 1
        """

        for rule_idx, delayed in enumerate(self.delayed_indicator_vct):
            if delayed:
                neuron_idx = self.__get_mapped_neuron(rule_idx)
                for rule in self.neuron_rule_map[neuron_idx]:
                    self.delay_status_vct[rule] -= 1

        for rule_idx, decision in enumerate(self.decision_vct):
            if decision:
                neuron_idx = self.__get_mapped_neuron(rule_idx)
                delay = self.rule_delay_vct[rule_idx]
                for rule in self.neuron_rule_map[neuron_idx]:
                    self.delay_status_vct[rule] = delay

    def compute_spikeable_mx(self):
        """
        Creates a matrix of all possible combinations of rules that can be activated
        """
        activatable_rules, activatable_count = self.get_activatable_rules()
        comb_count = np.prod(activatable_count, where=activatable_count > 0)
        self.spikeable_mx = np.zeros((int(comb_count), self.rule_count))
        temp_comb_count = comb_count

        for neuron_idx, _ in enumerate(self.neurons):
            if activatable_count[neuron_idx] == 0:
                continue

            rules = self.__get_active_rules(neuron_idx, activatable_rules)
            amount = temp_comb_count / activatable_count[neuron_idx]
            repeats = comb_count / temp_comb_count
            index_i = 0

            for _ in range(int(repeats)):
                for index_j in rules:
                    counter = 0
                    while counter < amount:
                        self.spikeable_mx[index_i, index_j] = 1
                        counter = counter + 1
                        index_i = index_i + 1

            temp_comb_count /= activatable_count[neuron_idx]

    def __get_active_rules(self, neuron, active):
        """
        Returns the indices of the active rules of a neuron
        """
        active = active.T[neuron]
        indices = [index for index, rule in enumerate(active) if rule]
        return indices

    def __init_delayed_indicator_vct(self):
        """
        Initializes the delayed spikeable vector to all zeros
        with a length equal to the number of rules
        """
        delayed_indicator_vct = np.zeros((self.rule_count,))
        return delayed_indicator_vct

    def __init_neuron_status_vct(self):
        """
        Initializes the neuron status vector to all ones
        with a length equal to the number of neurons
        """
        neuron_status_vct = np.ones((self.neuron_count,))
        return neuron_status_vct

    def __init_trans_mx(self):
        """
        Creates the Spiking Transition Matrix
        by parsing the rules provided
        """
        trans_mx = np.zeros(
            (
                self.rule_count,
                self.neuron_count,
            )
        ).astype(int)
        self.rule_delay_vct = np.zeros((self.rule_count,)).astype(int)

        for rule in self.rules:
            rule_idx = self.rule_keys.index(rule)
            self.rule_delay_vct[rule_idx] = self.rules[rule]["delay"]

            neuron_idx = self.__get_mapped_neuron(rule_idx)
            trans_mx[rule_idx] = np.dot(
                self.adj_mx[neuron_idx], self.rules[rule]["production"]
            )
            trans_mx[rule_idx, neuron_idx] = self.rules[rule]["consumption"]

        return trans_mx

    def __init_delay_status_vct(self):
        """
        Initializes the delay status vector to all zeros
        with a length equal to the number of rules
        """
        delay_status_vct = np.zeros((self.rule_count,)).astype(int)
        return delay_status_vct

    def __init_spike_train_vct(self):
        """
        Initializes the spike train vector to all zeros
        with a length equal to the number of neurons
        """
        spike_train_vct = np.array(
            [str(n.content) if n.type != "regular" else "" for n in self.neurons],
            dtype=object,
        ).astype(object)
        return spike_train_vct

    def __init_config_vct(self):
        """
        Creates a matrix of initial neuron configurations
        """
        # config_vct = np.array([n.content for n in self.neurons if n.type == 'regular' else n.spiketrain[:-1] if n.type == 'input' else 0])

        config_vct = np.array(
            [n.content if n.type == "regular" else 0 for n in self.neurons]
        )
        return config_vct

    def __set_neuron_order(self):
        """
        Creates a list of neuron keys in the order of their appearance
        """
        self.neuron_keys = [n.id for n in self.neurons]
        self.regular_keys = [
            neuron_idx
            for neuron_idx, n in enumerate(self.neurons)
            if n.type == "regular"
        ]
        self.input_keys = [
            neuron_idx for neuron_idx, n in enumerate(self.neurons) if n.type == "input"
        ]
        self.output_keys = [
            neuron_idx
            for neuron_idx, n in enumerate(self.neurons)
            if n.type == "output"
        ]
        self.neuron_count = len(self.neuron_keys)

    def __set_rule_order(self):
        """
        Creates a dictionary of rules in the order of their appearance
        """
        self.rule_count = 0
        self.rules: Dict[str, Dict[str, Union[str, int]]] = {}
        self.neuron_rule_map: Dict[int, list[int]] = {}
        for neuron_idx, n in enumerate(self.neurons):
            if n.type == "regular":
                self.neuron_rule_map[neuron_idx] = []
                for rule in n.rules:
                    new_neurons, bound, consumption, production, delay = parse_rule(rule)
                    self.rules[f"r{self.rule_count}"] = {
                        "new_neurons": new_neurons,
                        "bound": bound,
                        "consumption": consumption,
                        "production": production,
                        "type": "division" if new_neurons != (None, None) else "spiking" if production > 0 else "forgetting",
                        "delay": delay,
                    }
                    self.neuron_rule_map[neuron_idx].append(self.rule_count)
                    self.rule_count += 1
            elif n.type == "input":
                self.neuron_rule_map[neuron_idx] = []
                self.rules[f"r{self.rule_count}"] = {
                    "new_neurons": (None, None),
                    "bound": ".+",
                    "consumption": 0,
                    "production": 0,
                    "type": "input",
                    "delay": 0,
                }
                self.neuron_rule_map[neuron_idx].append(self.rule_count)
                self.rule_count += 1
            elif n.type == "output":
                self.neuron_rule_map[neuron_idx] = []
                self.rules[f"r{self.rule_count}"] = {
                    "new_neurons": (None, None),
                    "bound": ".+",
                    "consumption": 0,
                    "production": 0,
                    "type": "output",
                    "delay": 0,
                }
                self.neuron_rule_map[neuron_idx].append(self.rule_count)
                self.rule_count += 1

        self.rule_keys = list(self.rules.keys())

    def __init_adj_mx(self):
        """
        Creates an adjacency matrix of the neurons of the system
        """
        adj_mx = np.zeros((self.neuron_count, self.neuron_count)).astype(int)

        for synapse in self.synapses:
            if synapse.from_ in self.neuron_keys and synapse.to in self.neuron_keys:
                source = self.neuron_keys.index(synapse.from_)
                target = self.neuron_keys.index(synapse.to)
                adj_mx[source, target] = synapse.weight
        return adj_mx

    def check_non_determinism(self):
        """
        Checks if multiple rules are applicable in each neuron. Returns the list of neurons with non-determinism along with their rules.
        """
        non_deterministic_neurons = {}
        for neuron_idx, neuron in enumerate(self.neurons):
            if neuron.type == "regular":
                applicable_rules = []
                for rule_idx, _ in enumerate(neuron.rules):
                    rule = f"r{self.neuron_rule_map[neuron_idx][rule_idx]}"
                    bound: str = str(self.rules[rule]["bound"])
                    spikes: int = self.config_vct[neuron_idx]
                    neuron_status: int = self.neuron_status_vct[neuron_idx]
                    rule_validity = np.multiply(
                        neuron_status, check_rule_validity(bound, spikes)
                    ).astype(int)
                    if rule_validity:
                        applicable_rules.append(rule_idx)
                if len(applicable_rules) > 1:
                    non_deterministic_neurons[neuron.id] = applicable_rules
                elif len(applicable_rules) == 1:
                    non_deterministic_neurons[neuron.id] = applicable_rules[0]
        return non_deterministic_neurons

    def create_spiking_vector(self, choice: dict):
        spiking_vector = np.zeros((self.rule_count,)).astype(int)
        for neuron in choice:
            neuron_idx = self.neuron_keys.index(neuron)
            rule_idx = self.neuron_rule_map[neuron_idx][choice[neuron]]
            spiking_vector[rule_idx] = 1
        self.decision_vct = spiking_vector

    def get_activatable_rules(self):
        """
        Returns a matrix of rules that are activatable by the current configuration
        """
        activatable_rules_mx = np.zeros(
            (
                self.rule_count,
                self.neuron_count,
            )
        )
        active_rules_in_neuron = np.zeros((self.neuron_count,))
        for rule_idx, rule in enumerate(self.rules):
            neuron_idx = self.__get_mapped_neuron(rule_idx)

            bound: str = str(self.rules[rule]["bound"])
            spikes: int = self.config_vct[neuron_idx]
            neuron_status: int = self.neuron_status_vct[neuron_idx]
            rule_validity = np.multiply(
                neuron_status, check_rule_validity(bound, spikes)
            ).astype(int)
            activatable_rules_mx[rule_idx, neuron_idx] = rule_validity
            active_rules_in_neuron[neuron_idx] += rule_validity
        return activatable_rules_mx, active_rules_in_neuron

    def __get_mapped_neuron(self, rule_idx: int):
        neuron_idx = next(
            key for key, val in self.neuron_rule_map.items() if rule_idx in val
        )
        return neuron_idx

# Neuron division

    def __init_config_label_vct(self):
        """
        Initializes the config label vector to all zeros
        with a length equal to the number of neurons
        """
        config_label_vct = np.zeros((self.neuron_count,))
        return config_label_vct

    def __init_mask_vct(self):
        """
        Initializes the mask vector to all zeros
        with a length equal to the number of rules
        """
        mask_vct = np.zeros((self.rule_count,))
        return mask_vct
 
    def __init_checker_vct(self):
        """
        Initializes the checker vector to all zeros
        with a length equal to the number of rules
        """
        checker_vct = np.zeros((self.rule_count,)).astype(int)
        return checker_vct

    def __update_config_label_vct(self):
        """
        Updates the config label vector based on current iteration
        """
        self.config_label_vct = np.where(self.neuron_keys != None, self.neuron_keys, np.full((self.neuron_count,), -1)).astype(object)

    def __update_mask_vct(self):
        """
        Updates the mask vector based on current iteration
        """
        division_bool = [rules["type"] == "division" for rules in self.rules.values()]
        self.mask_vct = np.where(division_bool, np.full((self.rule_count,), 1), np.full((self.rule_count,), 0))

    def __update_checker_vct(self):
        """
        Updates the checker vector by performing elementwise multiplication on the spiking vector
        (i.e., the decision vector) and the mask vector
        """
        self.checker_vct = np.multiply(self.decision_vct, self.mask_vct).astype(int)

    def __reset_offset(self):
        """
        Reset the value of the offset for each iteration
        """
        self.offset = 0

    def __init_graphs(self):
        adj_lst = {self.neuron_keys[i]:[] for i in range(self.neuron_count)} 
        for i in range(self.neuron_count):
            for j in range(self.neuron_count):
                if self.adj_mx[i][j] != 0: adj_lst[self.neuron_keys[i]].append(self.neuron_keys[j])
        return [adj_lst]
    
    def __update_neuron_labels(self):
        self.neuron_labels.append(self.neuron_keys)

    def __update_graphs(self):
        adj_lst = {self.neuron_keys[i]:[] for i in range(self.neuron_count)} 
        for i in range(self.neuron_count):
            for j in range(self.neuron_count):
                if self.adj_mx[i][j] != 0: adj_lst[self.neuron_keys[i]].append(self.neuron_keys[j])
        self.graphs.append(adj_lst)
    
    def __rule_dict_parse(self):
        for neuron in self.neurons:
            if neuron.type == "regular":
                neuron.rules = []
                for rule in self.rule_dict:
                    if rule_dict_lookup(neuron.id, rule)[0]:
                        neuron.rules.append(rule_dict_lookup(neuron.id, rule)[1])

    def __divide_neuron(self, neuron_id, i, j):
        child1_new_rules = []
        child2_new_rules = []
        new_rules = 0
        child1_id = self.rules[f"r{i}"]['new_neurons'][0] 
        child2_id = self.rules[f"r{i}"]['new_neurons'][1]
        for rule in self.rule_dict:
            if rule_dict_lookup(child1_id, rule)[0]:
                child1_new_rules.append(rule_dict_lookup(child1_id, rule)[1])
                new_rules += 1
            if rule_dict_lookup(child2_id, rule)[0]:
                child2_new_rules.append(rule_dict_lookup(child2_id, rule)[1])
                new_rules += 1
        if len(child1_new_rules) == 0: child1_new_rules = ["a\\to \\lambda"]
        if len(child2_new_rules) == 0: child2_new_rules = ["a\\to \\lambda"]
        child1 = Regular(
            id = child1_id,
            type = "regular",
            content = 0,
            rules = child1_new_rules
        )
        child2 = Regular(
            id = child2_id,
            type = "regular",
            content = 0,
            rules = child2_new_rules
        )
        synapse_idx = 0
        while synapse_idx < len(self.synapses):
            synapse = self.synapses[synapse_idx]
            if synapse.from_ == neuron_id: 
                self.synapses.insert(synapse_idx, Synapse(
                    from_ = child1.id,
                    to = synapse.to,
                    weight = float(1)
                ))
                self.synapses[synapse_idx+1] = Synapse(
                    from_ = child2.id,
                    to = synapse.to,
                    weight = float(1)
                )
                synapse_idx -= 1
            if synapse.to == neuron_id: 
                self.synapses.insert(synapse_idx, Synapse(
                    from_ = synapse.from_,
                    to = child1.id,
                    weight = float(1)
                ))
                self.synapses[synapse_idx+1] = Synapse(
                    from_ = synapse.from_,
                    to = child2.id,
                    weight = float(1)
                )
                synapse_idx -= 1
            synapse_idx += 1
        
        for rule_idx in self.neuron_rule_map[j]:
            self.decision_vct[rule_idx] = 0
            self.delay_status_vct[rule_idx] = 0
            self.delayed_indicator_vct[rule_idx] = 0
            self.mask_vct[rule_idx] = 0
            self.checker_vct[rule_idx] = 0

        self.neurons.insert(j, child1)
        self.neurons[j+1] = child2

        self.config_vct = np.r_[self.config_vct[:j], [0, 0], self.config_vct[j+1:]]
        self.spike_train_vct = np.r_[self.spike_train_vct[:j], ['', ''], self.spike_train_vct[j+1:]]
        self.neuron_status_vct = np.r_[self.neuron_status_vct[:j], [0, 0], self.neuron_status_vct[j+1:]]
        self.config_label_vct = np.delete(self.config_label_vct, j)
        self.config_label_vct = np.insert(self.config_label_vct, j, [child1.id, child2.id])

        self.delay.insert(j, 0)
        self.delay[j+1] = 0

        old_rule_count = self.rule_count

        self.__set_neuron_order()
        self.__set_rule_order()

        if self.rule_count > old_rule_count:
            diff = self.rule_count - old_rule_count + 1
            self.decision_vct = np.r_[self.decision_vct[:i], np.zeros(diff), self.decision_vct[i+1:]]
            self.delay_status_vct = np.r_[self.delay_status_vct[:i], np.zeros(diff), self.delay_status_vct[i+1:]]
            self.delayed_indicator_vct = np.r_[self.delayed_indicator_vct[:i], np.zeros(diff), self.delayed_indicator_vct[i+1:]]
            self.mask_vct = np.r_[self.mask_vct[:i], np.zeros(diff), self.mask_vct[i+1:]]
            self.checker_vct = np.r_[self.checker_vct[:i], np.zeros(diff), self.checker_vct[i+1:]]
        else:
            diff = old_rule_count - self.rule_count + 1
            self.decision_vct = np.r_[self.decision_vct[:i], np.zeros(1), self.decision_vct[i+diff:]]
            self.delay_status_vct = np.r_[self.delay_status_vct[:i], np.zeros(1), self.delay_status_vct[i+diff:]]
            self.delayed_indicator_vct = np.r_[self.delayed_indicator_vct[:i], np.zeros(1), self.delayed_indicator_vct[i+diff:]]
            self.mask_vct = np.r_[self.mask_vct[:i], np.zeros(1), self.mask_vct[i+diff:]]
            self.checker_vct = np.r_[self.checker_vct[:i], np.zeros(1), self.checker_vct[i+diff:]]

        self.adj_mx = self.__init_adj_mx()
        self.trans_mx = self.__init_trans_mx()

        self.compute_spikeable_mx()
        self.__choose_decision_vct()

    def __division_algorithm(self):
        """
        Perform division algorithm
        """
        rule_idx = 0
        while rule_idx < len(self.checker_vct):
            if self.checker_vct[rule_idx] == 1:
                self.decision_vct[rule_idx] = 0
                neuron_idx = self.__get_mapped_neuron(rule_idx)
                neuron_id = self.neurons[neuron_idx].id
                self.__divide_neuron(neuron_id, rule_idx, neuron_idx + self.offset)
            rule_idx += 1