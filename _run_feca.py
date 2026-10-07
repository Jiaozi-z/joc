import sys, yaml
from recbole.quick_start import run
with open(sys.argv[1]) as f: cfg = yaml.safe_load(f)
run('FECA', cfg.pop('dataset'), config_file_list=[cfg.pop('base_config')], config_dict=cfg)
