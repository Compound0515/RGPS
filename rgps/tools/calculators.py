import torch


def get_calculator(model_type, model_path):
    """Factory to load MACE, NEP, or DeepMD calculators."""
    m_type = model_type.lower()

    if m_type in ["mace", "mace_mp"]:
        from mace.calculators import MACECalculator

        device = "cuda" if torch.cuda.is_available() else "cpu"
        return MACECalculator(model_paths=[model_path], device=device)

    elif m_type in ["nep", "cpunep"]:
        from calorine.calculators import CPUNEP

        return CPUNEP(model_path)

    elif m_type in ["dp", "deepmd"]:
        from deepmd.calculator import DP

        return DP(model=model_path)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
