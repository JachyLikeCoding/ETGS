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
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self._kernel_size = 0.1
        self.ray_jitter = True
        self.data_device = "cuda"
        self.scale_min = 0.0001
        self.scale_max = 0.02
        self.if_align = True
        self.eval = False
        super().__init__(parser, "Loading Model Parameters...", sentinel)
    
    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class DatasetParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.image_width = 0
        self.image_height = 0
        self.extent = 0
        self.volume_x = 0
        self.volume_y = 0
        self.volume_z = 0
        self.grid_size = 10
        self.sample_ratio = 0.1
        self.intensity_threshold = 0.02
        self.axis_id = 0 # x:0 y:1 z:2
        self.scale_factor = 1.0
        self.random_init = False
        super().__init__(parser, "Dataset Parameters...", sentinel)


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.02 #0.00016
        self.position_lr_final = 0.000001 #0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.scaling_lr = 0.05 #0.005
        self.scaling_lr_init = 0.05
        self.scaling_lr_final = 0.0005
        self.scaling_lr_delay_mult = 0.1
        self.scaling_lr_max_steps = 30_000
        self.rotation_lr = 0.05 #0.001
        self.rotation_lr_init = 0.05
        self.rotation_lr_final = 0.00005
        self.rotation_lr_delay_mult = 0.01
        self.rotation_lr_max_steps = 30_000
        self.embedding_lr = 0.005
        self.intensity_lr = 0.05
        self.intensity_lr_init = 0.05
        self.intensity_lr_final = 0.00005
        self.intensity_lr_delay_mult = 0.01
        self.intensity_lr_max_steps = 30_000
        self.deformation_lr_init = 0.00016
        self.deformation_lr_final = 0.000016
        self.deformation_lr_delay_mult = 0.01
        self.deformation_lr_max_steps = 30_000
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_tv = 0.05
        self.densification_interval = 100
        self.intensity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.densify_scale_threshold = 0.1
        self.random_background = False
        self.num_viewpoints = 1
        self.offsets_lr = 0.00002
        self.coef_tv_tilt_embedding = 0
        self.reg_coef = 1.0
        self.max_gaussians_num = 6_000_000
        self.tv_vol_size = 32
        
        super().__init__(parser, "Optimization Parameters")


class ModelHiddenParams(ParamGroup):
    def __init__(self, parser):
        self.net_width = 64
        self.deform_depth = 4
        self.min_embeddings = 30
        self.max_embeddings = 61
        self.use_deform = True
        self.no_ds = True
        self.no_dr = True
        self.no_di = True
        
        self.tilt_embedding_dim = 32
        self.gaussian_embedding_dim = 32
        self.use_coarse_tilt_embedding = False
        self.no_c2f_tilt_embedding = False
        self.no_coarse_deform = False
        self.no_fine_deform = False
        
        self.total_num_tilts = 41
        self.c2f_tilt_iter = 1000 # 20000
        self.deform_from_iter = 0
        self.use_anneal = True
        self.zero_tilt = False
        super().__init__(parser, "ModelHiddenParams")
        

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
