import argparse

from recbole.quick_start import run


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='FECA')
    parser.add_argument('--dataset', default='hdf-405')
    parser.add_argument('--config', dest='config_file_list', action='append')
    parser.add_argument('--ip', default='localhost')
    parser.add_argument('--port', default='5678')
    parser.add_argument('--world_size', type=int, default=-1)
    parser.add_argument('--nproc', type=int, default=1)
    parser.add_argument('--group_offset', type=int, default=0)
    args = parser.parse_args()
    if args.config_file_list is None:
        args.config_file_list = ['config/model_config.yaml']
    return args


if __name__ == "__main__":
    args = parse_args()

    run(
        args.model,
        args.dataset,
        args.config_file_list,
        nproc=args.nproc,
        world_size=args.world_size,
        ip=args.ip,
        port=args.port,
        group_offset=args.group_offset,
    )
