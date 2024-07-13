import re
from app.errors import WebsnapseError


def check_rule_validity(bound: str, spikes: int):
    """
    Performs regex matching to check
    if the number of spikes inside neuron
    satisfies the bound of the rule
    """
    if not spikes:
        return False
    bound = re.sub("\\^(\\d)", "^{\\1}", bound).replace("^", "")
    bound = re.sub(r"\{\s*\\ast\s*\}", "*", bound)
    bound = re.sub(r"\{\s*\*\s*\}", "*", bound)
    bound = re.sub(r"\{\s*\+\s*\}", "+", bound)
    bound = re.sub(r"\\ast", "*", bound)
    parsedBound = f"^{bound}$"
    validity = re.match(parsedBound, "a" * spikes)
    return validity is not None


def validate_rule(rule: str):
    pattern = r"^((?P<bound>.*)\/)?(?P<consumption_bound>[a-z](\^((?P<consumed_single>[^\D])|({(?P<consumed_multiple>[2-9]|[1-9][0-9]+)})))?)\s*(\\rightarrow|\\to)\s*(?P<production>([a-z]((\^((?P<produced_single>[^0,1,\D])|({(?P<produced_multiple>[2-9]|[1-9][0-9]+]*)})))?\s*;\s*(?P<delay>[0-9]|[1-9][0-9]*))|(?P<forgot>0)|(?P<lambda>\\lambda)))$"

    result = re.match(pattern, rule)

    if result is None:
        raise WebsnapseError(f"Invalid rule definition ${rule}$")

    return result


def parse_rule(definition: str):
    """
    Performs regex matching on the rule definition to get
    the consumption, production and delay values
    """
    result = validate_rule(definition)

    forgetting = True if result.group("forgot") or result.group("lambda") else False

    consumption = (
        result.group("consumed_multiple") or result.group("consumed_single") or 1
    )
    production = (
        result.group("produced_multiple") or result.group("produced_single") or 1
        if not forgetting
        else 0
    )
    delay = int(result.group("delay") or 1 if not forgetting else 0)

    consumption = -int(consumption)
    production = int(production)

    bound = result.group("bound") or result.group("consumption_bound")

    return bound, consumption, production, delay
