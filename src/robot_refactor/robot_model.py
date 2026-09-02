import pinocchio as pin
import numpy as np

from typing import Union, Optional, TypeAlias, Any
from pathlib import Path

FilePathType: TypeAlias = Union[Path, str]
ArrayLikeType: TypeAlias = Union[list[float], np.ndarray]
JointNameIdx: TypeAlias = list[str]

class JointAdapter:
    def __init__(
        self,
        # Pinocchio Model
        model: Any,
        # 输入 q pos 各个索引对应的 URDF 关节名称
        source_idx: Optional[JointNameIdx] = None,
        # 基于 Pinocchio 规则排列的默认关节角度, 未提供时使用 Pinocchio 配置空间的 neutral element
        q_pin_default: Optional[ArrayLikeType] = None
    ) -> None:

        self.n_q_pin = int(model.nq)
        
        if source_idx is not None:
            self.n_q_raw = len(source_idx)
            self.raw_in_pin_idx = np.zeros(self.n_q_raw, dtype = np.int32)
            for raw_idx, joint_name in enumerate(source_idx):

                if not model.existJointName(joint_name):
                    raise RuntimeError(f"Joint not found: {joint_name}")

                joint_id = model.getJointId(joint_name)

                # 一个 source joint 对应 q_raw 中的一个标量
                if model.nqs[joint_id] != 1:
                    raise RuntimeError(
                        f"Joint {joint_name!r} has nq= {model.nqs[joint_id]}, expected 1"
                        "JointAdapter currently supports scalar Pinocchio configurations only."
                    )

                self.raw_in_pin_idx[raw_idx] = model.idx_qs[joint_id]

        else:
            self.n_q_raw = self.n_q_pin
            self.raw_in_pin_idx = None

        if q_pin_default is None:
            # 获取 URDF 的自然关节角
            q_pin_default = pin.neutral(model) # type: ignore
        q_pin_default = np.asarray(q_pin_default, dtype = float)

        if q_pin_default.shape != (self.n_q_pin,):
            raise ValueError(
                f"Expected q_raw shape ({self.n_q_pin},), "
                f"got {q_pin_default.shape}"
            )
        self.q_pin_default = np.copy(q_pin_default)
                                      
    def to_q_pin(self, q_raw: ArrayLikeType):
        q_raw = np.asarray(q_raw, dtype = float)
        if q_raw.shape != (self.n_q_raw,):
            raise ValueError(
                f"Expected q_raw shape ({self.n_q_raw},), "
                f"got {q_raw.shape}"
            )

        if self.raw_in_pin_idx is not None:
            q_pin = np.copy(self.q_pin_default)
            q_pin[self.raw_in_pin_idx] = q_raw
            return q_pin
        else:
            return np.copy(q_raw)

    def to_q_raw(self, q_pin: np.ndarray):
        if q_pin.shape != (self.n_q_pin,):
            raise ValueError(
                f"Expected q_pin shape ({self.n_q_pin},), "
                f"got {q_pin.shape}"
            )

        if self.raw_in_pin_idx is not None:
            q_raw = q_pin[self.raw_in_pin_idx]
            return np.copy(q_raw)
        else:
            return np.copy(q_pin)

class RobotModel:

    def __init__(
        self,
        urdf_path: FilePathType,
        package_dir: Optional[FilePathType] = None,
        srdf_path: Optional[FilePathType] = None,

        # 输入 q pos 各个索引对应的 URDF 关节名称
        source_idx: Optional[JointNameIdx] = None,
        # 基于 Pinocchio 规则排列的默认关节角度, 未提供时使用 URDF 定义的自然关节位置
        q_pin_default: Optional[ArrayLikeType] = None,
    ):
        # =========================
        # Load model
        # =========================

        (self.model, self.collision_model) = pin.buildModelsFromUrdf( # type: ignore
            urdf_path,
            package_dirs = package_dir if package_dir is not None else [],
            geometry_types = pin.GeometryType.COLLISION, # type: ignore
        )

        # =========================
        # Collision pairs
        # =========================

        self.collision_model.addAllCollisionPairs()
        # 尝试自动获取 SRDF 文件
        if srdf_path is None:
            urdf_path = Path(urdf_path)
            srdf_path = urdf_path.with_suffix(".srdf")
            if not srdf_path.is_file():
                srdf_path = None
        # 加载 SRDF
        if srdf_path is not None:
            pin.removeCollisionPairs( # type: ignore
                self.model,
                self.collision_model,
                srdf_path,
            )

        # =========================
        # Runtime data
        # =========================

        self.data = self.model.createData()
        self.collision_data = (
            self.collision_model.createData()
        )

        # 关节数组转换器
        self.joint_adapter = JointAdapter(
            self.model, source_idx, q_pin_default
        )

        # 记录模型路径信息
        self.urdf_path = Path(urdf_path)
        self.srdf_path = Path(srdf_path) if srdf_path is not None else None
        self.package_dir = Path(package_dir) if package_dir is not None else None

    # --------------------------------
    # 实用函数
    # --------------------------------

    def get_q_neutral(self):
        '''
        获取默认关节角 (默认为自然关节角, 否则使用传入的默认关节角)
        '''
        return self.joint_adapter.to_q_raw(self.joint_adapter.q_pin_default)

    def get_generalized_gravity(self, q: ArrayLikeType):
        q = self.joint_adapter.to_q_pin(q)
        tau_g = pin.computeGeneralizedGravity( # type: ignore
            self.model,
            self.data,
            q
        )
        return self.joint_adapter.to_q_raw(np.asarray(tau_g))

    # --------------------------------
    # FK
    # --------------------------------

    def forward_kinematics(
        self,
        q: ArrayLikeType,
        frame_name: str,
    ):
        q = self.joint_adapter.to_q_pin(q)

        pin.forwardKinematics( # type: ignore
            self.model,
            self.data,
            q,
        )
        pin.updateFramePlacements( # type: ignore
            self.model,
            self.data,
        )

        frame_id = self.model.getFrameId(
            frame_name
        )

        tcp_T = self.data.oMf[frame_id]
        tcp_loc = np.asarray(tcp_T.translation)
        tcp_rot = np.asarray(tcp_T.rotation)
        return (tcp_loc, tcp_rot)

    # --------------------------------
    # Collision boolean
    # --------------------------------

    def is_self_collision(self, q: ArrayLikeType):
        q = self.joint_adapter.to_q_pin(q)

        return bool(
            pin.computeCollisions( # type: ignore
                self.model,
                self.data,
                self.collision_model,
                self.collision_data,
                q = q,
                stop_at_first_collision = True,
            )
        )

    # --------------------------------
    # All colliding pairs
    # --------------------------------

    def self_collision_pairs(self, q: ArrayLikeType):
        q = self.joint_adapter.to_q_pin(q)

        pin.computeCollisions( # type: ignore
            self.model,
            self.data,
            self.collision_model,
            self.collision_data,
            q = q,
            stop_at_first_collision = False,
        )

        output = []

        for pair_id, pair in enumerate(
            self.collision_model.collisionPairs
        ):
            result = (
                self.collision_data
                .collisionResults[pair_id]
            )

            if not result.isCollision():
                continue

            g1 = self.collision_model.geometryObjects[
                pair.first
            ]

            g2 = self.collision_model.geometryObjects[
                pair.second
            ]

            output.append(
                (g1.name, g2.name)
            )

        return output

    # --------------------------------
    # Minimum distance
    # --------------------------------

    def min_self_distance(
            self, 
            q: ArrayLikeType,
            threshold_distance: Optional[float] = None
        ):
        q = self.joint_adapter.to_q_pin(q)

        pin.computeDistances( # type: ignore
            self.model,
            self.data,
            self.collision_model,
            self.collision_data,
            q = q,
        )

        min_distance = float("inf")
        min_pair = None
        outer_pairs = []

        for pair_id, pair in enumerate(
            self.collision_model.collisionPairs
        ):
            result = (
                self.collision_data
                .distanceResults[pair_id]
            )
            d = result.min_distance

            cur_distance = d
            g1 = (
                self.collision_model
                .geometryObjects[pair.first]
            )
            g2 = (
                self.collision_model
                .geometryObjects[pair.second]
            )
            cur_pair = (
                g1.name,
                g2.name,
            )

            if cur_distance < min_distance:
                min_distance = cur_distance
                min_pair = cur_pair
            if threshold_distance is not None:
                if cur_distance < threshold_distance:
                    outer_pairs.append((cur_distance, cur_pair))

        return min_distance, min_pair, outer_pairs

def main(
    urdf_path: FilePathType = "./resources/robot.urdf",
    resource_dir: Optional[FilePathType] = "./resources",
    srdf_path: Optional[FilePathType] = None,
):
    '''
    测试将 URDF 加载为 Pinocchio 模型并输出机器人信息
    '''

    import time

    load_timer = time.perf_counter()
    rm = RobotModel(
        urdf_path, resource_dir, srdf_path, 
    )
    load_timer = time.perf_counter() - load_timer
    print(f"load urdf in {load_timer:.3f}s")
    joint_idx = {name:rm.model.idx_qs[rm.model.getJointId(name)] for name in rm.model.names}
    print(f"robot joint idx: {joint_idx}") 

    return rm

if __name__ == "__main__":
    import tyro
    tyro.cli(main)