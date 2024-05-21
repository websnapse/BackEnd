import re


def check_rule_validity(bound: str, spikes: int):
    """
    Performs regex matching to check
    if the number of spikes inside neuron
    satisfies the bound of the rule
    """
    if not spikes:
        return False
    bound = bound.replace("\\left", "").replace("\\right", "")
    bound = re.sub("\\^(\\d)", "^{\\1}", bound).replace("^", "")
    bound = re.sub(r"\{\s*\\ast\s*\}", "*", bound)
    bound = re.sub(r"\{\s*\*\s*\}", "*", bound)
    bound = re.sub(r"\{\s*\+\s*\}", "+", bound)
    bound = re.sub(r"\\ast", "*", bound)
    parsedBound = f"^{bound}$"
    validity = re.match(parsedBound, "a" * spikes)
    return validity is not None


def parse_rule(definition: str):
    """
    Performs regex matching on the rule definition to get
    the consumption, production and delay values
    """
    pattern = r"^(\\\bleft\b\[)?\s*((?P<bound>.*)\/)?(?P<consumption_bound>[a-z](\^((?P<consumed_single>[^\D])|({(?P<consumed_multiple>[2-9]|[1-9][0-9]+)})))?)\s*(\\\bright\b\])?\s*\\to\s*(?P<production>([a-z]((\^((?P<produced_single>[^0,1,\D])|({(?P<produced_multiple>[2-9]|[1-9][0-9]+]*)})))?(\s*;\s*(?P<delay>[0-9]|[1-9][0-9]*))?)|(?P<forgot>0)|(?P<lambda>\\lambda)|(?P<division>\\\bleft\b\[\s*\\\bright\b\]\_\{?(?P<new_neuron1>.+?)\}?\s*\\\bleft\b\|\\\bright\b\|\s*\\\bleft\b\[\s*\\\bright\b\]\_\{?(?P<new_neuron2>.+?)\}?)))$"

    result = re.match(pattern, definition)

    if result is None:
        return tuple((0, 0, 0, 0))

    forgetting = True if result.group("forgot") or result.group("lambda") else False

    consumption = (
        result.group("consumed_multiple") or result.group("consumed_single") or 1
    )
    production = (
        result.group("produced_multiple") or result.group("produced_single") or 1
        if not forgetting
        else 0
    )

    delay = 0
    if result.group("delay") != None: delay = int(result.group("delay") or 1 if not forgetting else 0)

    consumption = -int(consumption)
    production = int(production)

    bound = result.group("bound") or result.group("consumption_bound")

    new_neurons = (result.group("new_neuron1"), result.group("new_neuron2"))

    return new_neurons, bound, consumption, production, delay

def rule_dict_lookup(neuron_id: str, definition: str):
    """
    Performs regex matching from the rule dictionary to get
    the neuron ID and its rule in string form
    """

    pattern = r"^(?P<prefix>\\\bleft\b\[\s*(?P<rule>.+?)\s*\\\bright\b\])\_\{?(?P<neuron>.+?)\}?(?P<dynamic>\s*\\to\s*\\\bleft\b\[\s*\\\bright\b\]\_\{?.+?\}?\s*\\\bleft\b\|\\\bright\b\|\s*\\\bleft\b\[\s*\\\bright\b\]\_\{?.+?\}?)?$" 

    result = re.match(pattern, definition)

    if result.group("dynamic"):
        return neuron_id in result.group("neuron").split(","), result.group("prefix") + result.group("dynamic")

    return neuron_id in result.group("neuron").split(","), result.group("rule")