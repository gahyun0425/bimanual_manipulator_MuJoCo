# model 구동 노드

from __future__ import annotations
from typing import List, Optional

import rclpy
from rclpy.node import Node
import numpy as np

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState  

class BirrtTrajectoryExecutor(Node):

    def __init__(self):
        super().__init__('birrt_trajectory_executor')

        # ===== Parameters =====
        self.declare_parameter('left_joint_names', [
            "arm_l_joint1", "arm_l_joint2", "arm_l_joint3",
            "arm_l_joint4", "arm_l_joint5", "arm_l_joint6", "arm_l_joint7"
        ])

        self.declare_parameter('right_joint_names', [
            "arm_r_joint1","arm_r_joint2","arm_r_joint3",
            "arm_r_joint4","arm_r_joint5","arm_r_joint6","arm_r_joint7"
        ])

        self.declare_parameter('input_topic',   '/birrt/trajectory')
        self.declare_parameter('left_output_topic',  '/left_arm_controller/joint_trajectory')
        self.declare_parameter('right_output_topic',  '/right_arm_controller/joint_trajectory')

        self.declare_parameter('default_dt',    0.1)   # 입력에 time_from_start 없을 때 간격(초)
        
        self.declare_parameter('stream_topic',  '/desired_joint_angles')   # ★ 추가
        self.declare_parameter('stream_rate',   200.0)                    # ★ 추가 (Hz)

        self.left_joint_names: List[str] = list(self.get_parameter('left_joint_names').value)
        self.right_joint_names: List[str] = list(self.get_parameter('right_joint_names').value)
        
        self.input_topic: str  = self.get_parameter('input_topic').value
        self.left_out_topic: str  = self.get_parameter('left_output_topic').value
        self.right_out_topic: str = self.get_parameter('right_output_topic').value

        self.default_dt: float = float(self.get_parameter('default_dt').value)
        self.stream_topic: str = self.get_parameter('stream_topic').value      # ★
        self.stream_rate: float = float(self.get_parameter('stream_rate').value)# ★

        # ===== ROS IO =====
        self.pub_left  = self.create_publisher(JointTrajectory, self.left_out_topic, 10)
        self.pub_right = self.create_publisher(JointTrajectory, self.right_out_topic, 10)
        self.pub_js   = self.create_publisher(JointState,      self.stream_topic, 10)  # ★ JointState 퍼블리셔
        self.sub_planned = self.create_subscription(JointTrajectory, self.input_topic, self._on_birrt_trajectory, 10)

        # ★ JointState 스트리밍용 상태
        self._stream_traj: Optional[JointTrajectory] = None
        self._stream_times: Optional[np.ndarray] = None
        self._stream_start = None
        self._stream_timer = self.create_timer(1.0 / max(1.0, self.stream_rate), self._stream_tick)

        self.get_logger().info(f"[executor] left joints  = {self.left_joint_names}")
        self.get_logger().info(f"[executor] right joints = {self.right_joint_names}")
        self.get_logger().info(f"[executor] listen  : {self.input_topic}")
        self.get_logger().info(f"[executor] publish : L={self.left_out_topic} | R={self.right_out_topic}")
        self.get_logger().info(f"[executor] stream  : {self.stream_topic} @ {self.stream_rate}Hz")

    # ===== Core =====
    def _retime_if_needed(self, points: List[JointTrajectoryPoint]) -> List[float]:
        """입력 포인트들의 time_from_start가 0이거나 비증가하면 default_dt로 재타이밍."""
        use_default = False
        non_increasing = False
        last_t = -1.0
        for pt in points:
            t = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
            if pt.time_from_start.sec == 0 and pt.time_from_start.nanosec == 0:
                use_default = True
            if t <= last_t:
                non_increasing = True
            last_t = t

        retimed = []
        if use_default or non_increasing:
            t = 0.0
            for i, _ in enumerate(points):
                t += self.default_dt if i > 0 else max(0.1, self.default_dt)
                retimed.append(t)
        else:
            for pt in points:
                retimed.append(pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9)
        return retimed

    def _subset_and_publish(self, side: str, joint_names_out: List[str],
                            name_to_idx: dict, points_in: List[JointTrajectoryPoint],
                            retimed_times: List[float]) -> Optional[JointTrajectory]:
        """입력 궤적에서 지정된 조인트만 추출·리오더해 해당 팔 토픽에 퍼블리시."""
        # 해당 팔의 모든 조인트가 입력에 있어야 퍼블리시
        missing = [n for n in joint_names_out if n not in name_to_idx]
        if missing:
            self.get_logger().warn(f"[executor] skip {side}: missing joints {missing}")
            return None

        order = [name_to_idx[n] for n in joint_names_out]

        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = list(joint_names_out)

        for k, pt_in in enumerate(points_in):
            pt = JointTrajectoryPoint()
            # positions/vel/acc/effort 리오더(존재 시)
            if pt_in.positions:
                pt.positions = [float(pt_in.positions[i]) for i in order]
            if pt_in.velocities:
                pt.velocities = [float(pt_in.velocities[i]) for i in order]
            if pt_in.accelerations:
                pt.accelerations = [float(pt_in.accelerations[i]) for i in order]
            if pt_in.effort:
                pt.effort = [float(pt_in.effort[i]) for i in order]

            t = retimed_times[k]
            sec = int(t); nsec = int((t - sec) * 1e9)
            pt.time_from_start.sec = sec
            pt.time_from_start.nanosec = nsec
            traj.points.append(pt)

        # 퍼블리시
        if side == 'left':
            self.pub_left.publish(traj)
        else:
            self.pub_right.publish(traj)

        self.get_logger().info(
            f"[executor] published {side} traj: {len(traj.points)} points "
            f"(first={traj.points[0].positions if traj.points else []})"
        )
        return traj

    # ===== Core =====
    def _on_birrt_trajectory(self, msg_in: JointTrajectory):
        
        if not msg_in.points:
            self.get_logger().warn("[executor] input traj has 0 points; ignore.")
            return

        names_in = list(msg_in.joint_names)
        name_to_idx = {n: i for i, n in enumerate(names_in)}

        # 재타이밍(필요 시)
        retimed_times = self._retime_if_needed(msg_in.points)

        # 각 팔로 분리·퍼블리시
        left_traj  = self._subset_and_publish('left',  self.left_joint_names,
                                              name_to_idx, msg_in.points, retimed_times)
        right_traj = self._subset_and_publish('right', self.right_joint_names,
                                              name_to_idx, msg_in.points, retimed_times)

        if (left_traj is None) and (right_traj is None):
            self.get_logger().error("[executor] neither left nor right could be published (missing joints).")
        else:
            both = (left_traj is not None) and (right_traj is not None)
            self.get_logger().info(f"[executor] publish done. both_arms={both}")

        # ===== JointState 스트리밍 준비 (입력 joint 순서 그대로 유지) =====
        self._stream_traj = JointTrajectory()
        self._stream_traj.joint_names = list(msg_in.joint_names)
        self._stream_traj.points = []
        for k, pt in enumerate(msg_in.points):
            if not pt.positions:
                self.get_logger().error("[executor] input point has no positions. skip streaming.")
                self._stream_traj = None
                self._stream_times = None
                self._stream_start = None
                return
            p = JointTrajectoryPoint()
            p.positions = list(pt.positions)
            sec = int(retimed_times[k]); nsec = int((retimed_times[k] - sec) * 1e9)
            p.time_from_start.sec = sec
            p.time_from_start.nanosec = nsec
            self._stream_traj.points.append(p)

        self._stream_times = np.array(retimed_times, dtype=float)
        self._stream_start = self.get_clock().now()
        self.get_logger().info(f"[executor] start streaming JointState -> {self.stream_topic}")

    # ===== JointState 스트리밍 타이머 =====
    def _stream_tick(self):
        if self._stream_traj is None or self._stream_times is None or self._stream_start is None:
            return
        now = self.get_clock().now()
        t = (now - self._stream_start).nanoseconds * 1e-9

        pts = self._stream_traj.points
        times = self._stream_times

        if t <= times[0]:
            q = np.array(pts[0].positions, dtype=float)
        elif t >= times[-1]:
            q = np.array(pts[-1].positions, dtype=float)
        else:
            i = int(np.searchsorted(times, t, side='right')) - 1
            i = max(0, min(i, len(times) - 2))
            t0, t1 = times[i], times[i+1]
            s = (t - t0) / max(1e-9, (t1 - t0))
            q0 = np.array(pts[i].positions, dtype=float)
            q1 = np.array(pts[i+1].positions, dtype=float)
            q = (1.0 - s) * q0 + s * q1

        js = JointState()
        js.header.stamp = now.to_msg()
        js.name = list(self._stream_traj.joint_names)   # 입력 궤적의 이름/순서 유지 (7 또는 14)
        js.position = [float(v) for v in q]
        self.pub_js.publish(js)


def main(args=None):
    rclpy.init(args=args)
    node = BirrtTrajectoryExecutor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()