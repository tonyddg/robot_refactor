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
        if model.nq != model.nv:
            raise RuntimeError(
                "This RobotModel currently requires nq == nv"
            )

        if source_idx is not None:
            if len(source_idx) != len(set(source_idx)):
                raise ValueError(
                    "source_idx contains duplicate joint names"
                )

            self.n_q_raw = len(source_idx)
            self.raw_in_q_pin_idx = np.zeros(self.n_q_raw, dtype = np.int32)
            self.raw_in_v_pin_idx = np.zeros(self.n_q_raw, dtype=np.int32)

            for raw_idx, joint_name in enumerate(source_idx):

                if not model.existJointName(joint_name):
                    raise RuntimeError(f"Joint not found: {joint_name}")

                joint_id = model.getJointId(joint_name)
                nq = int(model.nqs[joint_id])
                nv = int(model.nvs[joint_id])

                # 各关节的 nq 数量等于 nv, 且一个 source joint 对应 q_raw 中的一个标量
                if nq != 1 or nv != 1:
                    raise RuntimeError(
                        f"Joint {joint_name!r} has "
                        f"nq={nq}, nv={nv}; expected nq=nv=1."
                    )

                self.raw_in_q_pin_idx[raw_idx] = model.idx_qs[joint_id]
                self.raw_in_v_pin_idx[raw_idx] = model.idx_vs[joint_id]
        else:
            self.n_q_raw = self.n_q_pin
            self.raw_in_q_pin_idx = None
            self.raw_in_v_pin_idx = None

        if q_pin_default is None:
            # 获取 URDF 的自然关节角
            q_pin_default = pin.neutral(model) # type: ignore
        q_pin_default = np.asarray(q_pin_default, dtype = float)

        if q_pin_default.shape != (self.n_q_pin,):
            raise ValueError(
                f"Expected q_pin shape ({self.n_q_pin},), "
                f"got {q_pin_default.shape}"
            )
        self.q_pin_default = np.copy(q_pin_default)
        self.v_pin_default = np.zeros(model.nv)
                                      
    def to_q_pin(self, q_raw: ArrayLikeType):
        q_raw = np.asarray(q_raw, dtype = float)
        if q_raw.shape != (self.n_q_raw,):
            raise ValueError(
                f"Expected q_raw shape ({self.n_q_raw},), "
                f"got {q_raw.shape}"
            )

        if self.raw_in_q_pin_idx is not None:
            q_pin = np.copy(self.q_pin_default)
            q_pin[self.raw_in_q_pin_idx] = q_raw
            return q_pin
        else:
            return np.copy(q_raw)

    def to_v_pin(self, v_raw: ArrayLikeType):
        v_raw = np.asarray(v_raw, dtype = float)
        if v_raw.shape != (self.n_q_raw,):
            raise ValueError(
                f"Expected v_raw shape ({self.n_q_raw},), "
                f"got {v_raw.shape}"
            )

        if self.raw_in_v_pin_idx is not None:
            q_pin = np.copy(self.v_pin_default)
            q_pin[self.raw_in_v_pin_idx] = v_raw
            return q_pin
        else:
            return np.copy(v_raw)

    def to_q_raw(self, q_pin: np.ndarray):
        if q_pin.shape != (self.n_q_pin,):
            raise ValueError(
                f"Expected q_pin shape ({self.n_q_pin},), "
                f"got {q_pin.shape}"
            )

        if self.raw_in_q_pin_idx is not None:
            q_raw = q_pin[self.raw_in_q_pin_idx]
            return np.copy(q_raw)
        else:
            return np.copy(q_pin)
        
    def to_v_raw(self, v_pin: np.ndarray):
        v_pin = np.asarray(v_pin, dtype=float)

        if v_pin.shape != (self.n_q_pin,):
            raise ValueError(
                f"Expected v_pin shape "
                f"({self.n_q_pin},), got {v_pin.shape}"
            )

        if self.raw_in_v_pin_idx is not None:
            return np.copy(
                v_pin[self.raw_in_v_pin_idx]
            )

        return np.copy(v_pin)

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
        '''
        共享的 Pinocchio 模型定义。

        被创建之后 model 与 collision_model 不可改变
        '''

        # =========================
        # Load model
        # =========================

        if package_dir is None:
            package_dirs: list[str] = []
        else:
            package_dirs = [str(package_dir)]

        (self.model, self.collision_model) = pin.buildModelsFromUrdf( # type: ignore
            urdf_path,
            package_dirs = package_dirs,
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

        # self.data = self.model.createData()
        # self.collision_data = (
        #     self.collision_model.createData()
        # )

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

    def get_q_default(self):
        '''
        获取默认关节角 (默认为自然关节角, 否则使用传入的默认关节角)
        '''
        return self.joint_adapter.to_q_raw(self.joint_adapter.q_pin_default)

    def get_frame_id(self, frame_name: str) -> int:
        '''
        获取 Frame ID
        '''
        if not self.model.existFrame(frame_name):
            raise ValueError(
                f"Frame not found: {frame_name!r}"
            )

        return int(self.model.getFrameId(frame_name))

    def create_context(self):
        '''
        可变的 Pinocchio 计算上下文。

        同一个 RobotContext 不应被多个线程并发访问；
        不同 RobotContext 可以并行计算。
        '''
        return RobotContext(self)

class RobotContext:

    def __init__(self, robot: RobotModel):
        '''
        可变的 Pinocchio 计算上下文。

        同一个 RobotContext 不应被多个线程并发访问；
        不同 RobotContext 可以并行计算。
        '''

        self.robot = robot
        self.data = robot.model.createData()
        self.collision_data = robot.collision_model.createData()

    def get_generalized_gravity(self, q: ArrayLikeType):
        '''
        计算机器人各个关节平衡重力所需的平衡力矩
        '''
        q = self.robot.joint_adapter.to_q_pin(q)
        tau_pin = pin.computeGeneralizedGravity( # type: ignore
            self.robot.model,
            self.data,
            q
        )
        return self.robot.joint_adapter.to_v_raw(np.asarray(tau_pin))

    # --------------------------------
    # FK
    # --------------------------------

    def forward_kinematics(
        self,
        q: ArrayLikeType,
        frame: Union[str, int],
    ):
        q = self.robot.joint_adapter.to_q_pin(q)

        pin.forwardKinematics( # type: ignore
            self.robot.model,
            self.data,
            q,
        )
        pin.updateFramePlacements( # type: ignore
            self.robot.model,
            self.data,
        )

        if isinstance(frame, str):
            frame_id = self.robot.get_frame_id(frame)
        else:
            frame_id = frame
            if not 0 <= frame_id < len(self.robot.model.frames):
                raise ValueError(
                    f"Invalid frame ID: {frame_id}"
                )

        tcp_T = self.data.oMf[frame_id]
        tcp_loc = np.array(tcp_T.translation, dtype = np.float64, copy = True)
        tcp_rot = np.array(tcp_T.rotation, dtype = np.float64, copy = True)
        return (tcp_loc, tcp_rot)

    # --------------------------------
    # Collision boolean
    # --------------------------------

    def is_self_collision(self, q: ArrayLikeType):
        q = self.robot.joint_adapter.to_q_pin(q)

        return bool(
            pin.computeCollisions( # type: ignore
                self.robot.model,
                self.data,
                self.robot.collision_model,
                self.collision_data,
                q = q,
                stop_at_first_collision = True,
            )
        )

    # --------------------------------
    # All colliding pairs
    # --------------------------------

    def self_collision_pairs(self, q: ArrayLikeType):
        q = self.robot.joint_adapter.to_q_pin(q)

        pin.computeCollisions( # type: ignore
            self.robot.model,
            self.data,
            self.robot.collision_model,
            self.collision_data,
            q = q,
            stop_at_first_collision = False,
        )

        output = []

        for pair_id, pair in enumerate(
            self.robot.collision_model.collisionPairs
        ):
            result = (
                self.collision_data
                .collisionResults[pair_id]
            )

            if not result.isCollision():
                continue

            g1 = self.robot.collision_model.geometryObjects[
                pair.first
            ]

            g2 = self.robot.collision_model.geometryObjects[
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
        q = self.robot.joint_adapter.to_q_pin(q)

        pin.computeDistances( # type: ignore
            self.robot.model,
            self.data,
            self.robot.collision_model,
            self.collision_data,
            q = q,
        )

        min_distance = float("inf")
        min_pair = None
        outer_pairs = []

        for pair_id, pair in enumerate(
            self.robot.collision_model.collisionPairs
        ):
            result = (
                self.collision_data
                .distanceResults[pair_id]
            )
            cur_distance = float(
                result.min_distance
            )
            g1 = (
                self.robot.collision_model
                .geometryObjects[pair.first]
            )
            g2 = (
                self.robot.collision_model
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