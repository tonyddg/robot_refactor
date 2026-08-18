from typing import Literal, Optional
from warnings import warn

import sapien
from sapien.wrapper.urdf_loader import URDFLoader
import numpy as np
from pathlib import Path
from transforms3d.quaternions import mat2quat
import time

from robot_refactor.robot_model import FilePathType, ArrayLikeType, RobotModel, JointNameIdx, JointAdapter

class RobotSim:
    def __init__(
        self,
        urdf_path: FilePathType,
        package_dir: Optional[FilePathType] = None,
        srdf_path: Optional[FilePathType] = None,
        # 基于 Pinocchio 规则排列的默认关节角度, 未提供时使用 URDF 定义的自然关节位置
        q_pin_default: Optional[ArrayLikeType] = None,
        # 外部输入 q pos (set_qpos 方法) 各个索引对应的 URDF 关节名称
        input_source_idx: Optional[JointNameIdx] = None,
        # 监视的 TCP 名称
        tcp_link_name: str = "left_tcp_link",
        # 仿真间隔
        timestep: float = 1e-2,
    ) -> None:
        urdf_path = urdf_path.resolve().as_posix() if isinstance(urdf_path, Path) else urdf_path 
        srdf_path = srdf_path.resolve().as_posix() if isinstance(srdf_path, Path) else srdf_path 
        package_dir = package_dir.resolve().as_posix() if isinstance(package_dir, Path) else package_dir 

        self.scene = sapien.Scene()
        self.scene.set_timestep(timestep)
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
        self.scene.add_ground(altitude = 0)

        # 加载机器人到 Sapien 中
        print("load sapien model")
        urdf_loader = URDFLoader()
        urdf_loader.set_scene(self.scene)
        urdf_loader.fix_root_link = True
        self.robot_object = urdf_loader.load(urdf_path, srdf_path, package_dir)
        self.robot_object.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))

        # 加载机器人 Pinocchio 模型
        print("load pinocchio model")
        source_idx = [
            joint_obj.name for joint_obj in self.robot_object.get_active_joints()
        ]
        self.robot_model = RobotModel(
            urdf_path, package_dir, srdf_path, source_idx = source_idx, q_pin_default = q_pin_default
        )
        self.input_joint_adapter = JointAdapter(
            self.robot_model.model, input_source_idx, q_pin_default
        )

        # 设置机器人控制器
        arm_zero_qpos = self.robot_model.get_q_neutral()
        active_joints = self.robot_object.get_active_joints()
        for joint_idx, joint in enumerate(active_joints):
            joint.set_drive_property(stiffness = 200, damping = 50, force_limit = 1000, mode="force")
            joint.set_drive_target(arm_zero_qpos[joint_idx])

        # 是否计算被动力平衡重力, 仅在开启动态时有效
        self.is_balance_passive_force = True
        # 是否计算场景动态 (否则仅依据 set_qpos 渲染场景)
        self.is_dynamic_scene = True

        # 检测的 TCP 名称
        self.tcp_link_name = tcp_link_name

        self.viewer = None

    def create_viewer(self):
        # 创建观察并设置界面
        self.viewer = self.scene.create_viewer()
        self.viewer.set_camera_xyz(x = -4, y = 0, z = 2)
        self.viewer.set_camera_rpy(r = 0, p = -np.arctan2(2, 4), y = 0)
        assert self.viewer.window is not None, "Fail to load viewer"
        self.viewer.window.set_camera_parameters(near = 0.05, far = 100, fovy = 1)

        self.scene.update_render()
        self.viewer.render()

    def _step_dynamic(self):
        # 总是跟踪最新的关节目标
        # if self.is_trace_cur_qpos:
        #     arm_zero_qpos = self.robot_object.get_qpos()
        #     active_joints = self.robot_object.get_active_joints()
        #     for joint_idx, joint in enumerate(active_joints):
        #         joint.set_drive_target(arm_zero_qpos[joint_idx])
        # 计算被动力并平衡
        if self.is_balance_passive_force:
            qf = self.robot_object.compute_passive_force(
                gravity = True,
                coriolis_and_centrifugal = True,
            )
            self.robot_object.set_qf(qf)
        else:
            qf = np.zeros_like(self.robot_object.get_qf())

        self.robot_object.set_qf(qf)
        # 更新场景物理参数
        self.scene.step()

    def _step_render(self):
        if self.viewer is None:
            return False
        if self.viewer.closed:
            return False

        self.scene.update_render()
        self.viewer.render()

        return True

    def _set_cur_as_target(self):
        arm_cur_qpos = self.robot_object.get_qpos()
        active_joints = self.robot_object.get_active_joints()
        for joint_idx, joint in enumerate(active_joints):
            joint.set_drive_target(arm_cur_qpos[joint_idx])
        print(f"set target to cur sapien qpos: {arm_cur_qpos}")

    def update_viewer(self):
        if self.viewer is None:
            # 外部关闭
            warn("Viewer is not created or has been closed")
            return False
        if self.viewer.closed:
            # 外部关闭
            return False
        assert self.viewer.window is not None, "Fail to load viewer"

        # 更新场景物理参数
        if self.is_dynamic_scene:
            self._step_dynamic()
        # 渲染场景
        is_open = self._step_render()
        if not is_open:
            return False

        # 开启或关闭被动力平衡
        if self.viewer.window.key_press("i"):
            self.is_balance_passive_force = not self.is_balance_passive_force
            print(f"is_balance_passive_force: {self.is_balance_passive_force}")
        # 将当前位置设置为动态控制目标
        elif self.viewer.window.key_press("o"):
            self._set_cur_as_target()
        # 切换场景物理动态
        elif self.viewer.window.key_press("p"):
            self.is_dynamic_scene = not self.is_dynamic_scene
            print(f"is_dynamic_scene: {self.is_dynamic_scene}")

            if self.is_dynamic_scene:
                # 从当前姿态开始动力学仿真
                self._set_cur_as_target()
            else:
                # 停止动力学残留状态
                self.robot_object.set_qf(np.zeros_like(self.robot_object.get_qf()))
                self.robot_object.set_qvel(np.zeros_like(self.robot_object.get_qvel()))
        
        # 获取关节角度
        elif self.viewer.window.key_press("x"):
            q_sapien = self.robot_object.get_qpos()
            q_pin = self.robot_model.joint_adapter.to_q_pin(q_sapien)
            q_input = self.input_joint_adapter.to_q_raw(q_pin)
            print(f"qpos sapien: {q_sapien}")
            print(f"qpos pinocchio: {q_pin}")
            print(f"qpos input {q_input}")
        # 正运动学
        elif self.viewer.window.key_press("c"):
            q_sapien = self.robot_object.get_qpos()
            tcp_loc, tcp_rot = self.robot_model.forward_kinematics(q_sapien, self.tcp_link_name)
            tcp_rot = mat2quat(tcp_rot)

            print(f"Pinocchio TCP FK result:\nloc: {tcp_loc}, rot: {tcp_rot}")
        # 碰撞检查
        elif self.viewer.window.key_press("v"):
            q_sapien = self.robot_object.get_qpos()

            collision_use_time = time.perf_counter()
            collision_info = self.robot_model.self_collision_pairs(q_sapien)
            collision_use_time = time.perf_counter() - collision_use_time

            distance_use_time = time.perf_counter()
            min_distance, min_pair, outer_pairs = self.robot_model.min_self_distance(q_sapien, 0.01)
            distance_use_time = time.perf_counter() - distance_use_time

            print(f"use time: {collision_use_time:.3f}, Pinnochio collision info: {collision_info}")
            print(f"use time: {distance_use_time:.3f}, Pinnochio min distance: {min_distance}, pair: {min_pair}\nouter_pairs: {outer_pairs}")

        elif self.viewer.window.key_down('q'):
            self.viewer.close()
            self.viewer = None
            return False

        return not self.viewer.closed

    def show_viewer(self):
        self.create_viewer()
        while True:
            is_open = self.update_viewer()
            if not is_open:
                break

    def set_state_qpos(
        self, 
        qpos: ArrayLikeType,
        # 取 False 认为是 Pinocchio 格式
        is_q_input: bool = True
    ):
        if self.is_dynamic_scene:
            warn(
                "set_state_qpos() is called while dynamic scene is enabled"
            )
        
        if is_q_input:
            q_input = qpos
            q_pin = self.input_joint_adapter.to_q_pin(q_input)
        else:
            q_pin = np.asarray(qpos)
        q_sapien = self.robot_model.joint_adapter.to_q_raw(q_pin)
        self.robot_object.set_qpos(q_sapien)

def main(
    urdf_path: FilePathType = "./resources/robot.urdf",
    resource_dir: Optional[FilePathType] = "./resources",
    srdf_path: Optional[FilePathType] = None,
    tcp_link_name: str = "left_tcp_link",
    # 基于 Pinocchio 规则排列的默认关节角度, 未提供时使用 URDF 定义的自然关节位置
    q_pin_default: Optional[list[float]] = None,
):
    rs = RobotSim(
        urdf_path, resource_dir, srdf_path, tcp_link_name = tcp_link_name, q_pin_default = q_pin_default
    )
    rs.show_viewer()

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
