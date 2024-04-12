def preprocess_scenario_filter(cfg): 
    """
    Function to modifiy a config so that when scenario_filter.scenario_tokens has modified tokens (e.g 5016a2a4ad1350d6-dmgl), 
    those get replaced by the base scenario (5016a2a4ad1350d6) and the modifications are added to scenario builder.
    This needs to be done since the nuplan scenario_filter_builder don't accept tokens with modifications
    """
    modifications = cfg.scenario_filter.modifications.scenario_specifics
    tokens = cfg.scenario_filter.scenario_tokens 
    
    modifications = modifications if modifications else []
    tokens = tokens if tokens else []

    # Assertions
    if any(map(lambda token: "-" in token, tokens)):
        assert all(map(lambda token: "-" in token, tokens)), \
        "Please enter either -> all scenario_filter.scenario_tokens in either base format (5016a2a4ad1350d6) and \
        change scenario_builder.modifications.scenario_specifics <- or -> all with modifications (5016a2a4ad1350d6-dmgl) <-"

    # Modify the config
    if modifications is not list or (modifications is list and len(tokens) != len(modifications)):

        modifications_dict = create_modifications_dictionary(tokens)

        cfg.scenario_filter.scenario_tokens = list(modifications_dict.keys())
        cfg.scenario_filter.modifications.scenario_specifics = list(modifications_dict.values())
        
def create_modifications_dictionary(tokens: list):
    modifications_dict = {}
    for token in tokens:
        if not "-" in token: token = token + "-"
        split = token.split("-")
        if split[0] not in modifications_dict: modifications_dict[split[0]] = []
        modifications_dict[split[0]].append(split[1])
    return modifications_dict