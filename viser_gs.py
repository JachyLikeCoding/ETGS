from threading import Thread
import torch
import numpy as np
import time
import viser
import viser.transforms as tf
from collections import deque
import math
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from plyfile import PlyData


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def world_to_camera(points, rot, tran):
    # r = torch.empty_like(points)
    # gaussian.world2camera(points, rot, tran, r)
    # return r
    # return world2camera_func(points, rot, tran)
    _r = points @ rot.T + tran.unsqueeze(0)
    return _r


def camera_to_image(points_camera_space):
    points_image_space = [
        points_camera_space[:, 0] / points_camera_space[:, 2],
        points_camera_space[:, 1] / points_camera_space[:, 2],
        points_camera_space.norm(dim=-1),
    ]
    return torch.stack(points_image_space, dim=-1)


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class MiniCam:
    def __init__(
        self,
        width,
        height,
        fovy,
        fovx,
        znear,
        zfar,
        world_view_transform,
        full_proj_transform,
        R,
        T,
    ):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        self.R = R
        self.T = T


# class MiniCam:
#     def __init__(
#         self,
#         width,
#         height,
#         fovy,
#         fovx,
#         znear,
#         zfar,
#         R,
#         T,
#         trans=np.array([0.0, 0.0, 0.0]),
#         scale=1.0,
#     ):
#         self.image_width = width
#         self.image_height = height
#         self.FoVy = fovy
#         self.FoVx = fovx
#         self.znear = znear
#         self.zfar = zfar
#         self.R = R
#         self.T = T

#         self.world_view_transform = (
#             torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
#         )
#         self.projection_matrix = (
#             getProjectionMatrix(
#                 znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
#             )
#             .transpose(0, 1)
#             .cuda()
#         )
#         self.full_proj_transform = (
#             self.world_view_transform.unsqueeze(0).bmm(
#                 self.projection_matrix.unsqueeze(0)
#             )
#         ).squeeze(0)
#         self.camera_center = self.world_view_transform.inverse()[3, :3]


@torch.no_grad()
def gs_render(
    viewpoint_camera: MiniCam,
    gs_xyz,
    gs_opacity,
    gs_scales,
    gs_rotations,
    gs_sh_features,
    bg_color,
    override_color,
    scaling_modifier=1.0,
):
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=3,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    screenspace_points = (
        torch.zeros_like(gs_xyz, dtype=gs_xyz.dtype, requires_grad=True, device="cuda")
        + 0
    )

    means3D = gs_xyz
    means2D = screenspace_points
    opacity = gs_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = gs_scales
    rotations = gs_rotations
    cov3D_precomp = None

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    shs = gs_sh_features

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    return rendered_image


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


class Splatter(torch.nn.Module):
    def __init__(self, device, config):
        super().__init__()
        self.device = device
        self.config = config
        self.bg_color = torch.tensor(
            config["bg_color"], dtype=torch.float32, device=self.device
        )

        self.scaling_activation = torch.exp

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.inverse_opacity_activation = inverse_sigmoid
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        self.read_gaussian_3ds(config["gaussian_model_ply_path"])

    def read_gaussian_3ds(self, path):
        self.max_sh_degree = 3
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["red"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["green"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["blue"])

        # extra_f_names = [
        #     p.name
        #     for p in plydata.elements[0].properties
        #     if p.name.startswith("f_rest_")
        # ]
        # extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        # assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        # features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        # for idx, attr_name in enumerate(extra_f_names):
        #     features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        # features_extra = features_extra.reshape(
        #     (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        # )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = torch.tensor(xyz, dtype=torch.float, device="cuda")
        self._features_dc = (
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
        )

        features = (
            torch.zeros((xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        )
        features[:, 3:, 1:] = 0.0

        self._features_rest = features[:, :, 1:].transpose(1, 2).contiguous()
        self._opacity = self.inverse_opacity_activation(
            1.0 * torch.ones((xyz.shape[0], 1), dtype=torch.float, device="cuda")
        )
        self._scaling = torch.tensor(scales, dtype=torch.float, device="cuda")
        self._rotation = torch.tensor(rots, dtype=torch.float, device="cuda")

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_features_rest(self):
        return self._features_rest

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def forward(self, extrinsics=None, intrinsics=None):

        view_cam = MiniCam(
            width=intrinsics["width"],
            height=intrinsics["height"],
            fovy=2 * math.atan(intrinsics["focal_y"] / 2) * 180 / math.pi,
            fovx=2 * math.atan(intrinsics["focal_x"] / 2) * 180 / math.pi,
            znear=0.1,
            zfar=1000.0,
            R=extrinsics["rot"],
            T=extrinsics["tran"],
        )

        render_img = gs_render(
            viewpoint_camera=view_cam,
            gs_xyz=self.get_xyz,
            gs_opacity=self.get_opacity,
            gs_scales=self.get_scaling,
            gs_rotations=self.get_rotation,
            gs_sh_features=self.get_features,
            bg_color=self.bg_color,
            override_color=None,
            scaling_modifier=1.0,
        )

        return render_img

    @torch.no_grad()
    def render_by_cam(self, view_cam):
        render_img = gs_render(
            viewpoint_camera=view_cam,
            gs_xyz=self.get_xyz,
            gs_opacity=self.get_opacity,
            gs_scales=self.get_scaling,
            gs_rotations=self.get_rotation,
            gs_sh_features=self.get_features,
            bg_color=self.bg_color,
            override_color=None,
            scaling_modifier=1.0,
        )

        return render_img


class Renderer:
    def __init__(self, device, config):
        self.device = device
        self.config = config

        self.gaussian_splatter = Splatter(device, config)

    @torch.no_grad()
    def test(self, view_cam, extrinsics=None, intrinsics=None):

        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)
        tic.record()
        # rendered_img = self.gaussian_splatter(extrinsics, intrinsics)
        rendered_img = self.gaussian_splatter.render_by_cam(view_cam)
        toc.record()
        torch.cuda.synchronize()
        render_time = tic.elapsed_time(toc) / 1000.0
        output = {"image": rendered_img, "render_time": render_time}
        return output


def get_c2w(camera):
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = qvec2rotmat(camera.wxyz)
    c2w[:3, 3] = camera.position
    return c2w


def get_w2c(camera):
    c2w = get_c2w(camera)
    w2c = np.linalg.inv(c2w)
    return w2c


class RenderThread(Thread):
    pass


class ViserViewer:
    def __init__(self, device, viewer_port):
        self.device = device
        self.port = viewer_port

        self.render_times = deque(maxlen=3)
        self.server = viser.ViserServer(port=self.port)
        self.reset_view_button = self.server.add_gui_button("Reset View")

        self.need_update = False

        self.pause_training = False
        self.train_viewer_update_period_slider = self.server.add_gui_slider(
            "Train Viewer Update Period",
            min=1,
            max=100,
            step=1,
            initial_value=10,
            disabled=self.pause_training,
        )

        self.resolution_slider = self.server.add_gui_slider(
            "Resolution", min=384, max=4096, step=2, initial_value=384
        )
        self.near_plane_slider = self.server.add_gui_slider(
            "Near", min=0.1, max=30, step=0.5, initial_value=0.1
        )
        self.far_plane_slider = self.server.add_gui_slider(
            "Far", min=30.0, max=1000.0, step=10.0, initial_value=100.0
        )

        self.fps = self.server.add_gui_text("FPS", initial_value="-1", disabled=True)

        @self.resolution_slider.on_update
        def _(_):
            self.need_update = True

        @self.near_plane_slider.on_update
        def _(_):
            self.need_update = True

        @self.far_plane_slider.on_update
        def _(_):
            self.need_update = True

        @self.reset_view_button.on_click
        def _(_):
            self.need_update = True
            for client in self.server.get_clients().values():
                client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array(
                    [0.0, -1.0, 0.0]
                )

        self.c2ws = []
        self.camera_infos = []

        @self.resolution_slider.on_update
        def _(_):
            self.need_update = True

        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            @client.camera.on_update
            def _(_):
                self.need_update = True

        self.debug_idx = 0

    def set_renderer(self, renderer):
        self.renderer = renderer

    @torch.no_grad()
    def update(self):
        if self.need_update:
            start = time.time()
            for client in self.server.get_clients().values():
                camera = client.camera
                w2c = get_w2c(camera)
                try:
                    W = self.resolution_slider.value
                    H = int(self.resolution_slider.value / camera.aspect)
                    fovx = camera.fov
                    fovy = camera.fov

                    start_cuda = torch.cuda.Event(enable_timing=True)
                    end_cuda = torch.cuda.Event(enable_timing=True)
                    start_cuda.record()

                    #########################################
                    c2w = torch.from_numpy(get_c2w(camera))
                    c2w_44 = torch.zeros(4, 4)
                    c2w_44[:3, :3] = c2w[:3, :3]
                    c2w_44[:3, 3] = c2w[:3, 3]
                    c2w_44[3, 3] = 1

                    c2w_44 = torch.inverse(c2w_44)

                    R = c2w[:3, :3].cpu().numpy()
                    T = c2w[:3, 3].cpu().numpy()

                    width = W
                    height = H

                    znear = self.near_plane_slider.value
                    zfar = self.far_plane_slider.value

                    world_view_transform = c2w_44.transpose(0, 1).cuda()

                    projection_matrix = (
                        getProjectionMatrix(
                            znear=znear, zfar=zfar, fovX=fovx, fovY=fovy
                        )
                        .transpose(0, 1)
                        .cuda()
                    )
                    full_proj_transform = (
                        world_view_transform.unsqueeze(0).bmm(
                            projection_matrix.unsqueeze(0)
                        )
                    ).squeeze(0)

                    full_proj_transform[:, 1] = -full_proj_transform[:, 1]
                    custom_cam = MiniCam(
                        width,
                        height,
                        fovy,
                        fovx,
                        znear,
                        zfar,
                        world_view_transform,
                        full_proj_transform,
                        R,
                        T,
                    )

                    outputs = self.renderer.test(
                        custom_cam
                        # extrinsics={
                        #     "rot": w2c[:3, :3],
                        #     "tran": w2c[:3, 3],
                        # },
                        # intrinsics={
                        #     "width": W,
                        #     "height": H,
                        #     "focal_x": focal_x,
                        #     "focal_y": focal_y,
                        # },
                    )
                    end_cuda.record()
                    torch.cuda.synchronize()
                    interval = start_cuda.elapsed_time(end_cuda) / 1000.0

                    out = outputs["image"].cpu().detach().numpy().astype(np.float32)
                    out = out.transpose(1, 2, 0)
                except RuntimeError as e:
                    print(e)
                    interval = 1
                    continue

                client.scene.set_background_image(out, format="jpeg")
                # # self.debug_idx += 1
                # # if self.debug_idx % 100 == 0:
                # #     cv2.imwrite(
                # #         f"./tmp/viewer/debug_{self.debug_idx}.png",
                # #         cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
                # #     )
                # import cv2

                # cv2.imshow("out", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
                # cv2.waitKey(1)

            self.render_times.append(interval)
            self.fps.value = f"{1.0 / np.mean(self.render_times):.3g}"
            end = time.time()
            # print(f"Update time: {end - start:.3g}")


if __name__ == "__main__":

    device = "cuda"

    renderer = Renderer(
        device=device,
        config={
            "bg_color": [0.0, 0.0, 0.0],
            "gaussian_model_ply_path": "/home/xin/Downloads/gaussian_points_rgb.ply",
        },
    )

    viewer = ViserViewer(device, 6789)
    viewer.set_renderer(renderer)
    while True:
        viewer.update()
        time.sleep(0.01)
    viewer.server.stop()
