from imgui_bundle import imgui
import torch
import numpy as np

from splatviz_utils.gui_utils.easy_imgui import label, slider, checkbox
from splatviz_utils.gui_utils import imgui_utils
from splatviz_utils.dict_utils import EasyDict
from splatviz_utils.cam_utils import (
    get_forward_vector,
    create_cam2world_matrix,
    get_origin,
    normalize_vecs,
)
from widgets.widget import Widget


class TiltWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "CryoET View")
        self.fov = 45
        self.radius = 3
        self.lookat_point = torch.tensor((0.0, 0.0, 0.0))
        self.forward = torch.tensor([0.0, -1.0, 0.0])
        self.tilt_angle = 0 # 当前倾斜角度
        self.cam_pos = torch.tensor([0.0, 0.0, 1.0])
        self.up_vector = torch.tensor([0.0, 0.0, 1.0]) # 固定Up方向为Z轴
        self.sample_bounds = torch.tensor([[-512, 512], [-512, 512], [-270, 270]]) # 样本范围

        # controls
        self.pose = EasyDict(yaw=0, pitch=0) # 仅用于视角调整
        self.move_speed = 0.02
        self.wasd_move_speed = 0.1
        self.drag_speed = 0.005
        self.rotate_speed = 0.02
        self.control_modes = ["Orbit", "WASD"]
        self.current_control_mode = 0
        self.last_drag_delta = imgui.ImVec2(0, 0)

    @imgui_utils.scoped_by_object_id
    def __call__(self, show: bool):
        viz = self.viz
        active_region = EasyDict(x=viz.pane_w, y=0, width=viz.content_width - viz.pane_w, height=viz.content_height)
        self.handle_dragging_in_window(**active_region)
        self.handle_mouse_wheel()
        self.handle_wasd()

        if show:
            label("Tilt Angle", viz.label_w)
            self.tilt_angle = slider(self.tilt_angle, "tilt_angle", -90, 90, format="%.1f°")

            label("Drag Speed", viz.label_w)
            self.drag_speed = slider(self.drag_speed, "drag_speed", 0.001, 0.1, log=True)

            label("Rotate Speed", viz.label_w)
            self.rotate_speed = slider(self.rotate_speed, "rot_speed", 0.001, 0.1, log=True)

            imgui.same_line()
            if imgui_utils.button("Reset view", width=viz.button_large_w):
                self.tilt_angle = 0
                self.pose.yaw = 0
                self.pose.pitch = 0
        self.update_view()


    def update_view(self):
        # 根据倾斜角度更新投影方向
        tilt_radians = np.deg2rad(self.tilt_angle)

        rotation_matrix = torch.tensor([
            [np.cos(tilt_radians), 0, np.sin(tilt_radians)],
            [0,1,0],
            [-np.sin(tilt_radians), 0, np.cos(tilt_radians)]
        ], dtype=torch.float32)

        self.forward = rotation_matrix @ torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
        self.cam_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32) # 固定为原点
        self.cam_params = create_cam2world_matrix(self.forward, self.cam_pos, self.up_vector)[0]
        # 更新渲染所需参数
        self.viz.args.forward = self.forward
        self.viz.args.up_vector = self.up_vector
        self.viz.args.tilt_angle = self.tilt_angle
        self.viz.args.cam_params = self.cam_params


    def handle_dragging_in_window(self, x, y, width, height):
        # 支持拖拽调整视角
        if imgui.is_mouse_dragging(0):  # left mouse button
            new_delta = imgui.get_mouse_drag_delta(0)
            if imgui_utils.did_drag_start_in_window(x, y, width, height, new_delta):
                delta = new_delta - self.last_drag_delta
                self.last_drag_delta = new_delta
                self.pose.yaw += delta.x * self.rotate_speed * 0.1
                self.pose.pitch += delta.y * self.rotate_speed * 0.1
                self.pose.pitch = np.clip(self.pose.pitch, -np.pi / 2, np.pi / 2)
        else:
            self.last_drag_delta = imgui.ImVec2(0, 0)


    def handle_wasd(self):
        if self.control_modes[self.current_control_mode] == "WASD":
            self.forward = get_forward_vector(
                lookat_position=self.cam_pos,
                horizontal_mean=self.pose.yaw + np.pi / 2,
                vertical_mean=self.pose.pitch + np.pi / 2,
                radius=0.01,
                up_vector=self.up_vector,
            )
            self.sideways = torch.linalg.cross(self.forward, self.up_vector)
            if imgui.is_key_down(imgui.Key.up_arrow) or "w" in self.viz.current_pressed_keys:
                self.cam_pos += self.forward * self.wasd_move_speed
            if imgui.is_key_down(imgui.Key.left_arrow) or "a" in self.viz.current_pressed_keys:
                self.cam_pos -= self.sideways * self.wasd_move_speed
            if imgui.is_key_down(imgui.Key.down_arrow) or "s" in self.viz.current_pressed_keys:
                self.cam_pos -= self.forward * self.wasd_move_speed
            if imgui.is_key_down(imgui.Key.right_arrow) or "d" in self.viz.current_pressed_keys:
                self.cam_pos += self.sideways * self.wasd_move_speed
            if "q" in self.viz.current_pressed_keys:
                self.cam_pos += self.up_vector * self.wasd_move_speed
            if "e" in self.viz.current_pressed_keys:
                self.cam_pos -= self.up_vector * self.wasd_move_speed

        elif self.control_modes[self.current_control_mode] == "Orbit":
            self.cam_pos = get_origin(
                self.pose.yaw + np.pi / 2,
                self.pose.pitch + np.pi / 2,
                self.radius,
                self.lookat_point,
                up_vector=self.up_vector,
            )
            self.forward = normalize_vecs(self.lookat_point - self.cam_pos)
            if imgui.is_key_down(imgui.Key.up_arrow) or "w" in self.viz.current_pressed_keys:
                self.pose.pitch += self.move_speed
            if imgui.is_key_down(imgui.Key.left_arrow) or "a" in self.viz.current_pressed_keys:
                self.pose.yaw += self.move_speed
            if imgui.is_key_down(imgui.Key.down_arrow) or "s" in self.viz.current_pressed_keys:
                self.pose.pitch -= self.move_speed
            if imgui.is_key_down(imgui.Key.right_arrow) or "d" in self.viz.current_pressed_keys:
                self.pose.yaw -= self.move_speed

    def handle_mouse_wheel(self):
        mouse_pos = imgui.get_io().mouse_pos
        if mouse_pos.x >= self.viz.pane_w:
            wheel = imgui.get_io().mouse_wheel
            if self.control_modes[self.current_control_mode] == "WASD":
                self.cam_pos += self.forward * self.move_speed * wheel
            elif self.control_modes[self.current_control_mode] == "Orbit":
                self.radius -= wheel / 10
