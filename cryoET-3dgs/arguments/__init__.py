from argparse import ArgumentParser, Namespace
import os, sys

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group    
    

class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.scene = ""
        self._source_path = "data/tomo"
        self._model_path = ""
        self._images = "images"
        self.resolution = -1
        self._white_background = False
        self._kernel_size = 0.1
        self.ray_jitter = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Model Parameters...", sentinel)
    
    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class DatasetParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.image_width = 1024
        self.image_height = 1024
        self.extent = 1024
        self.volume_x = 540
        self.volume_y = 540
        self.volume_z = 270
        self.grid_size = 10
        self.sample_ratio = 0.1
        self.intensity_threshold = 0.02
        self.axis_id = 0 # x:0 y:1 z:2
        super().__init__(parser, "Dataset Parameters...", sentinel)


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.0001 #0.00016
        self.position_lr_final = 0.000001 #0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 50_000
        self.scaling_lr = 0.05 #0.005
        self.rotation_lr = 0.05 #0.001
        self.intensity_lr = 0.05
        self.intensity_lr_init = 0.05
        self.intensity_lr_final = 0.00005
        self.intensity_lr_delay_mult = 0.01
        self.intensity_lr_max_steps = 50000
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 200
        self.intensity_reset_interval = 5000
        self.densify_from_iter = 500
        self.densify_until_iter = 50_000
        self.densify_grad_threshold = 0.0002
        self.random_background = False
        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
