from .plain_cnn import PlainCNN
from .res_cnn import ResCNN


def build_model(name: str):
    registry = {
        "plain": PlainCNN,
        "res": ResCNN,
    }
    try:
        return registry[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown model: {name}") from exc
