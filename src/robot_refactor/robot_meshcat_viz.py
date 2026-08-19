import pinocchio as pin

from typing import Optional
from robot_refactor.robot_model import RobotModel, FilePathType, ArrayLikeType

class RobotMeshcatViz:
    def __init__(
        self,
        robot_model: RobotModel,
        # 外部 meshcat 服务器地址, 通过 meshcat-cat 启动, 例如 "tcp://127.0.0.1:6000"
        zmq_url: Optional[str] = None,
        # 绘制坐标系的连杆或关节名称
        display_frame_names: Optional[list[str]] = None, 
    ) -> None:

        self.robot_model = robot_model
        
        self.visual_model = pin.buildGeomFromUrdf( # type: ignore
            model = self.robot_model.model,
            geom_type = pin.GeometryType.VISUAL, # type: ignore
            urdf_filename = self.robot_model.urdf_path,
            package_dirs = self.robot_model.package_dir if self.robot_model.package_dir is not None else [],
        )

        try:
            from pinocchio.visualize import MeshcatVisualizer
            self.viz = MeshcatVisualizer(
                model = self.robot_model.model, 
                collision_model = self.robot_model.collision_model, 
                visual_model = self.visual_model
            )
            self.viz.initViewer(open = False, zmq_url = zmq_url)
        except ImportError as err:
            raise ImportError("meshcat is not install, please install by pip robot_refactor[meshcat]")

        self.viz.loadViewerModel()
        if display_frame_names is not None:
            self._set_display_frame(display_frame_names)

        print(f"Visualizer open in url: {self.viz.viewer.url()}")

    def set_vis_qpos(
        self, 
        q: Optional[ArrayLikeType] = None,
    ):
        if q is None:
            q = self.robot_model.get_q_neutral()

        q_pin = self.robot_model.joint_adapter.to_q_pin(q)
        self.viz.display(q_pin)

    def _set_display_frame(
        self,
        # 绘制坐标系的连杆或关节名称
        display_frame_names: list[str],
    ):
        '''
        在初始化时调用, 不需要反复调用
        '''
        if self.viz is None:
            raise RuntimeError("to visualize robot, please set is_visualize_by_meshcat=True when create object")
        
        display_frame_ids = []
        for frame_name in display_frame_names:
            if not self.robot_model.model.existFrame(frame_name):
                raise RuntimeError(f"no frame {frame_name}")
            frame_id = self.robot_model.model.getFrameId(frame_name)
            display_frame_ids.append(frame_id)

        print(display_frame_ids)
        self.viz.displayFrames(
            True, frame_ids = display_frame_ids
        )

def main(
    urdf_path: FilePathType = "./resources/robot.urdf",
    resource_dir: Optional[FilePathType] = "./resources",
    srdf_path: Optional[FilePathType] = None,

    # 外部 meshcat 服务器地址, 通过 meshcat-cat 启动, 例如 "tcp://127.0.0.1:6000"
    zmq_url: Optional[str] = None,
    # 绘制坐标系的连杆或关节名称
    display_frame_names: Optional[list[str]] = None,
):
    '''
    测试将 URDF 加载为 Pinocchio 模型并输出机器人信息
    '''

    from robot_refactor.robot_model import main

    rm = main(
        urdf_path, resource_dir, srdf_path
    )
    rv = RobotMeshcatViz(
        rm, zmq_url, display_frame_names
    )
    rv.set_vis_qpos()

    return rv

if __name__ == "__main__":
    import tyro
    tyro.cli(main)