import click
import yaml
import os
import sys
from splatviz import Splatviz
from argparse import ArgumentParser, Namespace
from cryoET_3dgs.arguments import DatasetParams


@click.command()
@click.option("--data_path", help="root path for .ply files", metavar="PATH", 
              default="/media/feng/2480CDB880CD90AA/cryoET-reconstruction/cryoET-3DGS/output/")
@click.option("--mode", help="[default, decoder, attach]", default="default")
@click.option("--host", help="host address", default="127.0.0.1")
@click.option("--port", help="port", default=6009)
@click.option("--ggd_path", help="path to Gaussian GAN Decoder project", default="", type=click.Path())


def main(data_path, mode, host, port, ggd_path):
    parser = ArgumentParser(description="Training script parameters")
    dp = DatasetParams(parser)
    parser.add_argument('--config', type=str, default='config/10643.yaml', help='Path to the configuration file')
    args = parser.parse_args(sys.argv[1:])
    
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            setattr(args, key, value)
    
    data = dp.extract(args)
    print(f"[debug] args: {args}")
    print(f"[debug] data: {data}")
    splatviz = Splatviz(data_path=data_path, mode=mode, host=host, port=port, ggd_path=ggd_path, data=data)
    
    while not splatviz.should_close():
        splatviz.draw_frame()
    
    splatviz.close()


if __name__ == "__main__":
    main()
    #   default="/media/feng/2480CDB880CD90AA/cryoET-reconstruction/cryoET-3DGS/output/cryoet-10164/2024_11_30_17_01_24/gaussians/iteration_30000")
    #   default="/media/feng/2480CDB880CD90AA/cryoET-reconstruction/cryoET-3DGS/output/cryoet-10164/2024_11_27_17_22_46/gaussians/iteration_30000")
    #   default="/media/feng/2480CDB880CD90AA/cryoET-reconstruction/cryoET-3DGS/output/cryoet-10164/2024_11_27_17_22_46/gaussians/iteration_30000")
    #   default="/media/feng/2480CDB880CD90AA/cryoET-reconstruction/cryoET-3DGS/output/cryoet-10164/2025_01_12_18_45_35/gaussians/iteration_30000")
# /media/feng/2480CDB880CD90AA/cryoET-reconstruction/cryoET-3DGS/output/cryoet-10164/2025_01_14_21_21_27/gaussians/iteration_20000
# /media/feng/2480CDB880CD90AA/cryoET-reconstruction/cryoET-3DGS/output/cryoet-10164/2025_02_18_09_47_34/gaussians/iteration_20000