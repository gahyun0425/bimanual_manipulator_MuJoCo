# 장애물 랜덤 스폰 환경

from ament_index_python.packages import get_package_share_directory

import os
import time
import numpy as np
import mujoco
import mujoco.viewer

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


class MujocoSpawnerPublisher(Node):
    def __init__(self):
        super().__init__('mujoco_spawner_publisher')

        # === MuJoCo 로드 ===
        pkg_share = get_package_share_directory("new_bimanual_pkg")
        model_path = os.path.join(pkg_share, "mujoco_models", "bimanual.xml")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # object 바디의 freejoint qpos 인덱스 찾기
        object_body_id = self.model.body("object").id
        object_joint_id = self.model.body_jntadr[object_body_id]
        self.qpos_adr = int(self.model.jnt_qposadr[object_joint_id])  # [x,y,z,qw,qx,qy,qz]

        # === 테이블 위 랜덤 스폰 ===
        x = float(np.random.uniform(0.4, 0.6))
        y = float(np.random.uniform(-0.1, 0.1))
        z = 0.86
        self.data.qpos[self.qpos_adr:self.qpos_adr + 7] = [x, y, z, 1.0, 0.0, 0.0, 0.0]

        # === Publisher 생성 ===
        self.pub = self.create_publisher(PoseStamped, '/obstacle/pose', 10)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # === 1회성 타이머: 아주 짧게 뒤에 퍼블리시하고 종료 ===
        # (spin 전에 타이머가 준비될 수 있게 0.05s 지연)
        self._oneshot = self.create_timer(0.05, self._publish_and_quit)

        self.get_logger().info(
            f'Random obstacle set at ({x:.3f}, {y:.3f}, {z:.3f}). Will publish once and exit.'
        )

    def _publish_and_quit(self):
        # 더이상 이 콜백이 재호출되지 않도록 타이머 취소
        self._oneshot.cancel()

        # (선택) 구독자 붙을 때까지 잠깐 대기 — 최대 1초
        t0 = time.time()
        while self.pub.get_subscription_count() == 0 and time.time() - t0 < 1.0:
            time.sleep(0.02)

        # 현재 qpos에서 Pose 구성
        qpos = self.data.qpos[self.qpos_adr:self.qpos_adr + 7]
        x, y, z, qw, qx, qy, qz = [float(v) for v in qpos]

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'  # 필요시 프레임명 맞춰 변경
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.w = qw
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz

        self.pub.publish(msg)
        self.get_logger().info(
            f'Published random obstacle pose once: ({x:.3f}, {y:.3f}, {z:.3f})'
        )

        # DDS 전송 마무리 소폭 대기
        time.sleep(0.05)

        # 바로 종료
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = MujocoSpawnerPublisher()
    try:
        # _publish_and_quit()에서 shutdown 호출하면 여기서 빠져나옴
        rclpy.spin(node)
    finally:
        node.destroy_node()
