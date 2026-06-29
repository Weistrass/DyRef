def Flux2KleinTextEncoderStateDictConverter(state_dict):
    """
    State dict converter for Flux2KleinTextEncoder (Qwen3-8B based).
    
    Converts HuggingFace state dict format to the expected format.
    The main task is to handle potential key renaming from HF checkpoint format.
    """
    state_dict_ = {}
    for k in state_dict:
        k_ = k
        # Handle potential key renaming if loading from different formats
        # Qwen3 models typically have straightforward key names
        state_dict_[k_] = state_dict[k]
    return state_dict_
