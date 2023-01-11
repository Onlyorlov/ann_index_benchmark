from attrdict import AttrDict
import argparse
import yaml

from src.runner import run


def main(opt):
    with open(opt.config_pth) as f:
        config = yaml.safe_load(f)

    runs_params = config["runs"]
    for params in runs_params:
        run_cfg = {
            k: v for k, v in config.items() if k != "runs"
        }  # remove unnecessary param?
        for k, v in params.items():  # add run params
            run_cfg[k] = v
        cfg = AttrDict(run_cfg)
        print(f"Running {cfg.experiment_name} exp for {cfg.name} index")
        run(cfg.experiment_name, cfg)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_pth",
        help="Config file with multiple runs parameters",
        type=str,
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
