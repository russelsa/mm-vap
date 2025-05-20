import yaml
from turn_taking.model.model import StereoTransformerModel


if __name__ == "__main__":

    cfg = yaml.safe_load(open("turn_taking/assets/config.yaml", "r"))
    model = StereoTransformerModel(cfg=cfg)
