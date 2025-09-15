# MuJoCo에 모델 스폰

import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from ament_index_python.packages import get_package_share_directory
import mujoco
import mujoco.viewer

class MujocoROSNode(Node):
    def __init__(self, model, data):
        super().__init__('mujoco_node')
        self.model = model
        self.data = data

        # joint 이름 → qpos 인덱스 매핑
        self.joint_name_to_qposaddr = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j): model.jnt_qposadr[j]
            for j in range(model.njnt)
        }

        self.subscription = self.create_subscription(
            JointState,
            '/desired_joint_angles',
            self.joint_callback,
            10
        )
        self.get_logger().info('Subscribed to /desired_joint_angles')

    def joint_callback(self, msg: JointState):
        for name, position in zip(msg.name, msg.position):
            if name in self.joint_name_to_qposaddr:
                addr = self.joint_name_to_qposaddr[name]
                self.data.qpos[addr] = position
        mujoco.mj_forward(self.model, self.data)


def main():
    rclpy.init()

    # 모델 로딩
    pkg_path = get_package_share_directory('new_bimanual_pkg')
    model_path = os.path.join(pkg_path, 'mujoco_models', 'bimanual.xml')
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    node = MujocoROSNode(model, data)

    # 뷰어 실행
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while rclpy.ok() and viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            rclpy.spin_once(node, timeout_sec=0.001)

    node.destroy_node()
    rclpy.shutdown()
    print("[INFO] Viewer closed.")