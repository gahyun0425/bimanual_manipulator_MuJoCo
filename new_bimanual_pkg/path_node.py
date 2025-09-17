# master

import rclpy
import numpy as np
import pinocchio as pin
import time
import math

from rclpy.node import Node
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.srv import GetPositionIK
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from urdf_parser_py.urdf import URDF
import yaml, os

from new_bimanual_pkg.birrt import BiRRT
from new_bimanual_pkg.trajectory import plan_trajectory, build_spline, make_joint_trajectory_msg, clamp_eval
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy

def load_joint_limits(urdf_path: str, joint_limits_yaml: str, joint_names: list):
    # URDF에서 설정한 joint_limit 값으로 lb, ub 설정

    # URDF 파싱
    with open(urdf_path, 'r') as f:
        urdf_xml = f.read()
    robot = URDF.from_xml_string(urdf_xml)

    # URDF에서 읽은 limit 설정
    limits = {}
    for j in robot.joints:
        if j.type == 'continuous':
            limits[j.name] = (-np.pi, np.pi)
        elif j.type in ('revolute', 'prismatic'):
            limits[j.name] = (float(j.limit.lower), float(j.limit.upper))

    if joint_limits_yaml and os.path.exists(joint_limits_yaml):
        with open(joint_limits_yaml, 'r') as f:
            cfg = yaml.safe_load(f) or {}
        jlim = cfg.get('joint_limits', {})
        for name, v in jlim.items():
            has = v.get('has_position_limits', False)
            if has:
                limits[name] = (float(v['min_position']), float(v['max_position']))
            else:
                # has_position_limits: false 인 경우 연속 관절처럼 처리
                limits.setdefault(name, (-np.pi, np.pi))

    # 순서 정렬해서 배열 생성 (없으면 연속 관절처럼 fallback)
    lb = []
    ub = []
    for nm in joint_names:
        lo, hi = limits.get(nm, (-np.pi, np.pi))
        lb.append(lo)
        ub.append(hi)
    return np.array(lb, dtype=float), np.array(ub, dtype=float)

# RPY -> quaternion 
def quat_from_rpy(roll, pitch, yaw):
    """RPY(rad) -> quaternion [x,y,z,w] (world/arm_base_link 기준)"""
    cr = math.cos(roll*0.5);  sr = math.sin(roll*0.5)
    cp = math.cos(pitch*0.5); sp = math.sin(pitch*0.5)
    cy = math.cos(yaw*0.5);   sy = math.sin(yaw*0.5)
    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy
    return [qx, qy, qz, qw]

class PathNode(Node):
    def __init__(self):
        super().__init__('path_node')

        # --- gripper: MuJoCo direct-publish mode ---
        self.use_mujoco_grip = True  # 컨트롤러 없으니 True
        self.mj_grip_topic   = '/mujoco/gripper_set'  # 너의 브리지 입력 토픽명으로 변경
        self.mj_grip_pub     = self.create_publisher(JointState, self.mj_grip_topic, 10)


        # IK service client
        self.ik_cli = self.create_client(GetPositionIK, '/compute_ik')
        if not self.ik_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn("'/compute_ik' service not available yet; waiting in background.")


        # pinocchio FK -> 시각화용. IK보다 먼저 준비
        urdf_path = "/home/gaga/bimanual_ws/src/new_bimanual_pkg/mujoco_models/bimanual.urdf"
        full_model = pin.buildModelFromUrdf(urdf_path)

        # EE
        self.ee_frame_l = "gripper_l_rh_p12_rn_base"
        self.ee_frame_r = "gripper_r_rh_p12_rn_base"

        # left, right joint name
        left_names  = ['arm_l_joint1','arm_l_joint2','arm_l_joint3','arm_l_joint4','arm_l_joint5','arm_l_joint6','arm_l_joint7']
        right_names = ['arm_r_joint1','arm_r_joint2','arm_r_joint3','arm_r_joint4','arm_r_joint5','arm_r_joint6','arm_r_joint7']
        self.left_names  = left_names
        self.right_names = right_names
        self.joint_names = left_names + right_names 

        # 두 팔 별 퍼블리셔 (컨트롤러 토픽에 맞게 바꿔도 됨)
        self.traj_pub_left  = self.create_publisher(JointTrajectory, '/birrt/trajectory_left',  10)
        self.traj_pub_right = self.create_publisher(JointTrajectory, '/birrt/trajectory_right', 10)
        
        # 양팔 joint만 남기고 잠그기
        keep = set(self.joint_names)
        lock_ids = []
        for name in full_model.names[1:]:
            if name not in keep:
                jid = full_model.getJointId(name)
                if jid != 0:
                    lock_ids.append(jid)

        q0_full = pin.neutral(full_model)
        self.model = pin.buildReducedModel(full_model, lock_ids, q0_full)
        self.data = self.model.createData()

        # FK에서 사용할 EE
        self.ee_frame_id_l = self.model.getFrameId(self.ee_frame_l)
        self.ee_frame_id_r = self.model.getFrameId(self.ee_frame_r)

        # 기본은 왼팔로 시각화
        self.ee_frame_id = self.ee_frame_id_l

        # 시작자세 (RRT 시작점 및 FK용)
        self.start_q = np.zeros(len(self.joint_names), dtype=float)

        # 좌/우 목표 pose 입력
        print("\n왼팔 목표 위치 입력 (x y z) [예: 0.5 0.2275 1.199] : ", end="")
        raw = input()
        x, y, z = map(float, raw.strip().split())
        target_l = np.array([x, y, z], dtype=float)

        print("\n왼팔 방위 (roll pitch yaw in deg) [예: 90 0 0] : ", end="")
        raw = input()
        rL, pL, yL = map(float, raw.strip().split())
        rL = math.radians(rL); pL = math.radians(pL); yL = math.radians(yL)
        quat_l = quat_from_rpy(rL, pL, yL)

        print("\n오른팔 목표 위치 입력 (x y z) [예: 0.5 -0.2275 1.199] : ", end="")
        raw = input()
        x, y, z = map(float, raw.strip().split())
        target_r = np.array([x, y, z], dtype=float)

        print("\n오른팔 방위 (roll pitch yaw in deg) [예: -90 0 0] : ", end="")
        raw = input()
        rR, pR, yR = map(float, raw.strip().split())
        rR = math.radians(rR); pR = math.radians(pR); yR = math.radians(yR)
        quat_r = quat_from_rpy(rR, pR, yR)

        seed_left  = [0.0] * len(left_names)
        seed_right = [0.0] * len(right_names)

        base_frame = 'arm_base_link' 

        ik_l = self.compute_ik_via_moveit(
            group='left_arm',
            frame_id=base_frame,
            ik_link_name=self.ee_frame_l,    
            pos=target_l.tolist(),
            quat=quat_l,
            seed_names=left_names,
            seed_values=seed_left,
            timeout=0.5,
            attempts=10,
            avoid_collisions=True
        )
        if ik_l is None:
            raise RuntimeError("Left IK failed (TRAC-IK)")

        ik_r = self.compute_ik_via_moveit(
            group='right_arm',
            frame_id=base_frame,
            ik_link_name=self.ee_frame_r,    
            pos=target_r.tolist(),
            quat=quat_r,
            seed_names=right_names,
            seed_values=seed_right,
            timeout=0.5,
            attempts=10,
            avoid_collisions=True
        )
        if ik_r is None:
            raise RuntimeError("Right IK failed (TRAC-IK)")
        
        # moveit group & joint limit
        self.group_name = 'manipulator'
        
        joint_limits_yaml = '/home/gaga/bimanual_ws/src/bimanual_moveit_config/config/joint_limits.yaml'
        self.lb, self.ub = load_joint_limits(urdf_path, joint_limits_yaml, self.joint_names)

        # collision planner
        self.planner = BiRRT(
            joint_names=self.joint_names,
            lb=self.lb,
            ub=self.ub,
            group_name=self.group_name,
            state_dim=len(self.joint_names),
            max_iter=2000,
            step_size=0.03,
            edge_check_res=0.05,
        )

        # IK 결과를 goal로 결합
        self.goal_q = np.array([*(ik_l[nm] for nm in left_names),
                                *(ik_r[nm] for nm in right_names)], dtype=float)

        self.planner.set_start(self.start_q)
        self.planner.set_goal(self.goal_q)

        # publisher & marker qos
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=100,
        )

        self.marker_pub = self.create_publisher(Marker, '/visualization_marker', qos)
        self.path_pub_full = self.create_publisher(Path, '/birrt_path/full', qos)
        self.path_pub_a = self.create_publisher(Path, '/birrt_path/A', qos)
        self.path_pub_b = self.create_publisher(Path, '/birrt_path/B', qos)

        self.traj_topic = '/birrt/trajectory'
        self.traj_pub = self.create_publisher(JointTrajectory, self.traj_topic, 10)

        # 주기적으로 한 번만 계획/시각화
        self.timer = self.create_timer(1.0, self.plan_once_and_visualize)
        self.done = False

        # --- 도달 판정 상태 ---
        self._awaiting_arrival = False     # 트젝 보낸 뒤 true로, 도달 후 false
        self._arm_goal = {}                # {joint_name: 목표각}
        self._arrive_count = 0             # 연속 만족 카운트
        self._arrive_needed = 10           # 몇 번 연속 만족하면 도달로 볼지 (100Hz면 0.1s)

        # 토픽에서 최신 JointState 수신
        self.js_sub = self.create_subscription(
            JointState, '/joint_states', self._on_joint_state, 50
        )

        # 그리퍼 타깃 퍼블리셔(/gripper_target로 보냄)
        self.grip_target_pub = self.create_publisher(JointState, '/gripper_target', 10)

        # 램프 상태 멤버
        self._grip_ramp_timer = None
        self._grip_ramp_t0 = None
        self._grip_ramp_T = 1.0      # 총 램프 시간(초) – 취향대로
        self._grip_ramp_rate = 50.0  # 발행 주기(Hz)
        self._grip_start = {}
        self._grip_goal  = {}


        # --- add this in __init__ ---
        self.desired_pub = self.create_publisher(JointState, '/desired_joint_angles', 10)


    # ee frame의 quaternion 값 반환
    def fk_quat_at(self, q: np.ndarray, ee_frame: str):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        fid = self.model.getFrameId(ee_frame)
        T = self.data.oMf[fid]
        quat = pin.Quaternion(T.rotation).coeffs()  # [x,y,z,w]
        return [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
    
    # TRAK IK 호출
    def compute_ik_via_moveit(self, group, frame_id, ik_link_name, pos, quat,
                          seed_names, seed_values, timeout=0.2, attempts=8, avoid_collisions=False):
        
        # 서비스 대기
        if not self.ik_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("'/compute_ik' service not available")
            return None

        # 안전 캐스팅
        px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
        qx, qy, qz, qw = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])

        # 시드는 리스트로 작업
        seed_names = list(seed_names)
        base_seed  = [float(v) for v in seed_values]

        for k in range(max(1, int(attempts))):
            req = GetPositionIK.Request()
            req.ik_request.group_name = group
            req.ik_request.ik_link_name = ik_link_name
            req.ik_request.avoid_collisions = bool(avoid_collisions)

            ps = PoseStamped()
            ps.header.frame_id = frame_id            # ← MoveIt planning frame과 맞추세요 (보통 robot_description의 base)
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.pose.position.x = px
            ps.pose.position.y = py
            ps.pose.position.z = pz
            ps.pose.orientation.x = qx
            ps.pose.orientation.y = qy
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw
            req.ik_request.pose_stamped = ps

            # 시드 (k>0일 때 약간 섞어서 재시도)
            if k == 0:
                seed = base_seed
            else:
                jitter = np.random.normal(scale=1e-3, size=len(base_seed))
                seed = (np.array(base_seed) + jitter).tolist()

            req.ik_request.robot_state.joint_state = JointState()
            req.ik_request.robot_state.joint_state.name = seed_names
            req.ik_request.robot_state.joint_state.position = seed

            # ROS2는 timeout만 지원(필수)
            req.ik_request.timeout = Duration(
                sec=int(timeout),
                nanosec=int((timeout - int(timeout)) * 1e9)
            )

            future = self.ik_cli.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            res = future.result()
            if res is None:
                self.get_logger().error("IK service call failed (no response)")
                return None

            if res.error_code.val == res.error_code.SUCCESS:
                names = list(res.solution.joint_state.name)
                vals  = list(res.solution.joint_state.position)
                return dict(zip(names, vals))

            # 실패 로그는 첫/마지막 시도에만 간단히
            if k == 0 or k == attempts - 1:
                self.get_logger().warn(f"IK attempt {k+1}/{attempts} failed (error_code={res.error_code.val})")

            # 짧게 쉬고 재시도(선택)
            time.sleep(0.01)

        return None
    
    # FK (joint space -> 3D point for current ee_frame_id)
    def q_to_point(self, q: np.ndarray) -> Point:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        pose = self.data.oMf[self.ee_frame_id]
        p = Point()
        p.x, p.y, p.z = pose.translation.tolist()
        return p

    def _base_marker(self, mid: int, mtype: int, ns: str) -> Marker:
        m = Marker()
        m.header.frame_id = 'world'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = mid
        m.type = mtype
        m.action = Marker.ADD
        m.lifetime.sec = 0
        m.pose.orientation.w = 1.0
        return m

    def q_to_posestamped(self, q: np.ndarray) -> PoseStamped:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        T = self.data.oMf[self.ee_frame_id]
        R = T.rotation
        t = T.translation
        quat = pin.Quaternion(R).coeffs()  # [x,y,z,w]

        ps = PoseStamped()
        ps.header.frame_id = 'world'
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = t.tolist()
        ps.pose.orientation.x = float(quat[0])
        ps.pose.orientation.y = float(quat[1])
        ps.pose.orientation.z = float(quat[2])
        ps.pose.orientation.w = float(quat[3])
        return ps

    def make_polyline_marker(self, points, mid: int, ns:str, color:str, width:float):
        m = self._base_marker(mid, Marker.LINE_STRIP, ns)
        m.scale.x = width
        m.color.a = 1.0
        if color == 'r': m.color.r = 1.0
        elif color == 'g': m.color.g = 1.0
        elif color == 'b': m.color.b = 1.0
        else: m.color.r = m.color.g = 0.8
        m.points = points
        return m

    def make_tree_nodes_marker(self, points, mid: int, ns: str, color='g'):
        m = self._base_marker(mid, Marker.POINTS, ns)
        m.scale.x = 0.01; m.scale.y = 0.01
        m.color.a = 1.0
        if color == 'g':   m.color.g = 0.9
        elif color == 'b': m.color.b = 0.9
        else:              m.color.r = 0.9
        m.points = points
        return m

    def make_tree_edges_marker(self, edges, mid: int, ns: str, color='g'):
        m = self._base_marker(mid, Marker.LINE_LIST, ns)
        m.scale.x = 0.003
        m.color.a = 1.0
        if color == 'g':   m.color.g = 0.7
        elif color == 'b': m.color.b = 0.7
        else:              m.color.r = 0.7
        m.points = [pt for seg in edges for pt in seg]  # 두 점씩 한 선
        return m

    def make_sphere_marker(self, point, mid: int, ns: str, color='y'):
        m = self._base_marker(mid, Marker.SPHERE, ns)
        m.scale.x = m.scale.y = m.scale.z = 0.05
        m.color.a = 1.0
        if color == 'y': m.color.r, m.color.g = 0.9, 0.9
        elif color == 'g': m.color.g = 1.0
        elif color == 'b': m.color.b = 1.0
        else: m.color.r = 1.0
        m.pose.position = point
        return m

    def tree_to_points_edges(self, tree):
        pts, edges = [], []
        for n in tree:
            pts.append(self.q_to_point(n['q']))
        for idx, n in enumerate(tree):
            p = n['parent']
            if p is None:
                continue
            pa = self.q_to_point(tree[p]['q'])
            ch = self.q_to_point(n['q'])
            edges.append((pa, ch))
        return pts, edges

    def plan_once_and_visualize(self):
        if self.done:
            return
        self.done = True

        self.get_logger().info('Planning (Bi-RRT)...')
        result = self.planner.solve(max_time=5.0)

        ok = False; full = None
        path_a = getattr(self.planner, 'last_path_a', None)
        path_b = getattr(self.planner, 'last_path_b', None)
        if isinstance(result, tuple):
            if len(result) == 4:
                ok, full, path_a, path_b = result
            elif len(result) == 2:
                ok, full = result

        if not ok or full is None:
            self.get_logger().warning('Failed to find path.')
            return

        # ---------- 여기부터 추가 (기존 self.publish_joint_trajectory(...) 삭제/대체) ----------
        # full shape: (N, 14)  ← N: 웨이포인트 수
        dof_each = 7
        full = np.asarray(full)
        Q_L = full[:, :dof_each]               # (N,7) 왼팔
        Q_R = full[:, dof_each:2*dof_each]     # (N,7) 오른팔

        # 팔 별 속도/가속 상한 (필요시 파라미터화 가능)
        qd_max_L  = np.array([1.5]*dof_each)
        qdd_max_L = np.array([3.0]*dof_each)
        qd_max_R  = np.array([1.5]*dof_each)
        qdd_max_R = np.array([3.0]*dof_each)

        # TOPP→스플라인→균일샘플 (왼팔)
        outL = plan_trajectory(
            Q_L, qd_max_L, qdd_max_L,
            ds=0.005, sdot_start=0.0,
            stop_window_s=0.05,     # 끝에서만 감속
            alpha_floor=1.5,        # 중앙부 속도 바닥(상한의 20%)
            v_min_time=1e-4,
            sample_hz=10.0, max_points=100000
        )

        # TOPP→스플라인→균일샘플 (오른팔)
        outR = plan_trajectory(
            Q_R, qd_max_R, qdd_max_R,
            ds=0.005, sdot_start=0.0,
            stop_window_s=0.05,
            alpha_floor=1.5,
            v_min_time=1e-4,
            sample_hz=10.0, max_points=100000
        )

        # JointTrajectory 메시지로 변환 & 퍼블리시 (두 팔 따로)
        trajL = make_joint_trajectory_msg(
            joint_names=self.left_names,
            t0=float(outL["t_samples"][0]),
            t_samples=outL["t_samples"],
            Q_samples=outL["Q_samples"],
            Qd_samples=outL["Qd_samples"],
        )
        trajR = make_joint_trajectory_msg(
            joint_names=self.right_names,
            t0=float(outR["t_samples"][0]),
            t_samples=outR["t_samples"],
            Q_samples=outR["Q_samples"],
            Qd_samples=outR["Qd_samples"],
        )

        # --- 도달 판정을 위한 목표각 기록 & 감시 시작 ---
        goal_map = {}
        if trajL and len(trajL.points) > 0:
            lastL = trajL.points[-1].positions
            for nm, q in zip(self.left_names, lastL):
                goal_map[nm] = float(q)

        if trajR and len(trajR.points) > 0:
            lastR = trajR.points[-1].positions
            for nm, q in zip(self.right_names, lastR):
                goal_map[nm] = float(q)

        self._arm_goal = goal_map
        self._awaiting_arrival = True
        self._arrive_count = 0
        self.get_logger().info(f'Arrival monitoring ON for {len(goal_map)} joints.')


        now = self.get_clock().now().to_msg()
        if trajL is not None:
            trajL.header.frame_id = 'world'
            trajL.header.stamp = now
            self.traj_pub_left.publish(trajL)
        if trajR is not None:
            trajR.header.frame_id = 'world'
            trajR.header.stamp = now
            self.traj_pub_right.publish(trajR)

        self.get_logger().info(
            f"Published: left({len(trajL.points) if trajL else 0}) right({len(trajR.points) if trajR else 0})"
        )

        controller_hz = 100.0  # ← 실제 joint_trajectory_controller 주기
        dt = 1.0/controller_hz

        splineL = build_spline(outL["t_knots"], Q_L, outL["qdot_knots"])
        splineR = build_spline(outR["t_knots"], Q_R, outR["qdot_knots"])

        t0L, t1L = float(outL["t_samples"][0]), float(outL["t_samples"][-1])
        t0R, t1R = float(outR["t_samples"][0]), float(outR["t_samples"][-1])
        t0 = min(t0L, t0R); t1 = max(t1L, t1R)
        time_scale = 1.0  # 필요시 0.8 등으로 조절
        T = (t1 - t0) * time_scale

        t_common = t0 + np.arange(0.0, T + 0.5*dt, dt)

        # ===== 여기에 추가 =====
        QL_sync, QLd_sync, QR_sync, QRd_sync = [], [], [], []
        for t in t_common:
            qL, qLd = clamp_eval(splineL, t)
            qR, qRd = clamp_eval(splineR, t)
            QL_sync.append(qL);  QLd_sync.append(qLd)
            QR_sync.append(qR);  QRd_sync.append(qRd)

        QL_sync  = np.vstack(QL_sync)
        QLd_sync = np.vstack(QLd_sync)
        QR_sync  = np.vstack(QR_sync)
        QRd_sync = np.vstack(QRd_sync)

        Q_sync  = np.hstack([QL_sync,  QR_sync])   # (M,14)
        Qd_sync = np.hstack([QLd_sync, QRd_sync])  # (M,14)

        traj_all = make_joint_trajectory_msg(
            joint_names=(self.left_names + self.right_names),
            t0=t0,
            t_samples=(t_common - t0),   # time_from_start 기준이면 0부터
            Q_samples=Q_sync,
            Qd_samples=Qd_sync
        )
        if traj_all is not None:
            traj_all.header.frame_id = 'world'
            traj_all.header.stamp = self.get_clock().now().to_msg()
            self.traj_pub.publish(traj_all)

        # FULL PATH
        full_points = [self.q_to_point(q) for q in full]
        if len(full_points) >= 2:
            self.marker_pub.publish(self.make_polyline_marker(full_points, 0, 'birrt_path_full', 'r', 0.01))
            path_msg = Path()
            path_msg.header.frame_id = 'world'
            path_msg.header.stamp = self.get_clock().now().to_msg()
            path_msg.poses = [self.q_to_posestamped(q) for q in full]
            self.path_pub_full.publish(path_msg)

        # PARTIAL PATHS A/B
        def publish_partial(name, arr, mid, color, pub):
            if arr is None or len(arr) < 2:
                return
            pts = [self.q_to_point(q) for q in arr]
            self.marker_pub.publish(self.make_polyline_marker(pts, mid, name, color, 0.02))
            msg = Path()
            msg.header.frame_id = 'world'
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.poses = [self.q_to_posestamped(q) for q in arr]
            pub.publish(msg)

        publish_partial('birrt_path_A', path_a, 10, 'g', self.path_pub_a)
        publish_partial('birrt_path_B', path_b, 11, 'b', self.path_pub_b)

        # 트리 마커
        try:
            start_pts, start_edges = self.tree_to_points_edges(self.planner.start_tree)
            goal_pts,  goal_edges  = self.tree_to_points_edges(self.planner.goal_tree)

            self.marker_pub.publish(self.make_tree_nodes_marker(start_pts, 1, 'start_tree_nodes', 'g'))
            self.marker_pub.publish(self.make_tree_edges_marker(start_edges, 2, 'start_tree_edges', 'g'))
            self.marker_pub.publish(self.make_tree_nodes_marker(goal_pts,  3, 'goal_tree_nodes',  'b'))
            self.marker_pub.publish(self.make_tree_edges_marker(goal_edges, 4, 'goal_tree_edges',  'b'))
        except Exception as e:
            self.get_logger().warn(f"Tree visualization skipped: {e}")

        # 시작/목표 위치
        self.marker_pub.publish(self.make_sphere_marker(self.q_to_point(self.start_q), 5, 'start_pose', 'g'))
        self.marker_pub.publish(self.make_sphere_marker(self.q_to_point(self.goal_q),  6, 'goal_pose',  'b'))

        len_a = len(path_a) if path_a is not None else 0
        len_b = len(path_b) if path_b is not None else 0
        self.get_logger().info(f'Path found! full={len(full)} A={len_a} B={len_b} (markers published)')

    def publish_joint_trajectory(self, q_path: np.ndarray, dt: float = 0.1):
        if q_path is None or len(q_path) == 0:
            self.get_logger().warn("No path to publish.")
            return
        dof = len(self.joint_names)
        if q_path.shape[1] != dof:
            self.get_logger().error(f"Path DOF({q_path.shape[1]}) != joint_names({dof})")
            return

        traj = JointTrajectory()
        traj.header.frame_id = 'world'
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = list(self.joint_names)

        t = 0.0
        for q in q_path:
            pt = JointTrajectoryPoint()
            pt.positions = [float(x) for x in q]
            t += dt
            pt.time_from_start.sec = int(t)
            pt.time_from_start.nanosec = int((t - int(t)) * 1e9)
            traj.points.append(pt)

        self.traj_pub.publish(traj)
        self.get_logger().info(f"Published JointTrajectory to {self.traj_topic} ({len(traj.points)} points)")

    def close_grippers(self):
        names = [
            'gripper_r_joint1','gripper_r_joint2','gripper_r_joint3','gripper_r_joint4',
            'gripper_l_joint1','gripper_l_joint2','gripper_l_joint3','gripper_l_joint4'
        ]
        targets = [1.1,1.0,1.1,1.0, 1.1,1.0,1.1,1.0]

        msg = JointState()
        msg.name = names
        msg.position = targets

        # 퍼블리셔: /desired_joint_angles
        # __init__ 에서 self.desired_pub = self.create_publisher(JointState, '/desired_joint_angles', 10)
        if not hasattr(self, "_grip_hold_timer"):
            self._grip_hold_timer = None
        self._grip_hold_count = 0

        if self._grip_hold_timer:
            self._grip_hold_timer.cancel()
            self._grip_hold_timer = None

        def _tick():
            msg.header.stamp = self.get_clock().now().to_msg()
            self.desired_pub.publish(msg)
            self._grip_hold_count += 1
            if self._grip_hold_count >= 100:  # 100회 ≈ 0.5s @200Hz
                self._grip_hold_timer.cancel()
                self._grip_hold_timer = None

        self._grip_hold_timer = self.create_timer(1.0/200.0, _tick)
        
    def _on_joint_state(self, msg: JointState):
        # 도달 감시 중이 아니면 패스
        if not self._awaiting_arrival or not self._arm_goal:
            return

        # 관심 조인트만 인덱스 매핑
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        missing = [n for n in self._arm_goal.keys() if n not in name_to_idx]
        if missing:
            # 아직 모든 조인트가 joint_states에 안 뜨면 다음 메시지에서 다시 시도
            return

        # tol 설정(라디안 / 라디안/초)
        pos_tol = 0.01    # 0.57° 정도
        vel_tol = 0.02    # 관성/노이즈 고려해 약간 여유
        # JointState.velocity 길이가 0이거나 일부만 있으면 속도 판정 생략 가능
        have_vel = (len(msg.velocity) == len(msg.name) and len(msg.velocity) > 0)

        max_pos_err = 0.0
        max_abs_vel = 0.0

        for jn, q_goal in self._arm_goal.items():
            idx = name_to_idx[jn]
            q_cur = msg.position[idx]
            err = abs(q_cur - q_goal)
            if err > max_pos_err:
                max_pos_err = err
            if have_vel:
                v = abs(msg.velocity[idx])
                if v > max_abs_vel:
                    max_abs_vel = v

        # 판정: 위치가 충분히 가깝고(필수), 속도도 충분히 작으면(있으면) OK
        pos_ok = (max_pos_err <= pos_tol)
        vel_ok = (not have_vel) or (max_abs_vel <= vel_tol)

        if pos_ok and vel_ok:
            self._arrive_count += 1
        else:
            self._arrive_count = 0

        # 연속 만족 시 그리퍼 닫고 감시 종료
        if self._arrive_count >= self._arrive_needed:
            self._awaiting_arrival = False
            self.get_logger().info(
                f'Arrival confirmed: max_pos_err={max_pos_err:.4f}, max_abs_vel={max_abs_vel:.4f}'
            )
            # 서서히 닫기: 1.5초 동안, 60Hz로
            self.start_gripper_ramp(T=2.0, rate_hz=60.0)

    def _ease_smoothstep(self, s: float) -> float:
        # 0~1 -> 0~1, 가감속이 부드러운 프로파일
        s = max(0.0, min(1.0, s))
        return s*s*(3.0 - 2.0*s)

    def _grip_ramp_tick(self):
        # 진행률
        now = self.get_clock().now()
        s = (now - self._grip_ramp_t0).nanoseconds * 1e-9 / max(1e-6, self._grip_ramp_T)
        s = max(0.0, min(1.0, s))
        s_ease = self._ease_smoothstep(s)

        # 현재 보간 값 만들기 (원 코드)
        names = list(self._grip_goal.keys())
        pos = []
        for n in names:
            q0 = self._grip_start.get(n, 0.0)
            q1 = self._grip_goal[n]
            pos.append(q0 + (q1 - q0) * s_ease)

        # 퍼블리시 (원 코드)
        msg = JointState()
        msg.header.stamp = now.to_msg()
        msg.name = names
        msg.position = [float(v) for v in pos]
        self.grip_target_pub.publish(msg)

        # 종료 처리 (원 코드)
        if s >= 1.0 and self._grip_ramp_timer is not None:
            self._grip_ramp_timer.cancel()
            self._grip_ramp_timer = None
            self.get_logger().info("Gripper ramp done.")


    def start_gripper_ramp(self, goal: dict = None, T: float = 1.0, rate_hz: float = 50.0, start_from: dict = None):
        """
        goal: {'joint_name': target, ...}
        T: 램프 총 시간(초)
        rate_hz: 발행 주기(Hz)
        start_from: 시작값(없으면 0.0으로 가정)
        """
        # 기본 목표(2,4번은 1.0, 나머지 1.1)
        if goal is None:
            goal = {
                'gripper_r_joint1': 0.9, 'gripper_r_joint2': 0.9,
                'gripper_r_joint3': 0.9, 'gripper_r_joint4': 0.9,
                'gripper_l_joint1': 0.9, 'gripper_l_joint2': 0.9,
                'gripper_l_joint3': 0.9, 'gripper_l_joint4': 0.9,
            }

        # 시작값 없으면 0.0에서 시작(완전 오픈 가정)
        if start_from is None:
            start_from = {n: 0.0 for n in goal.keys()}

        # 상태 저장
        self._grip_start = {n: float(start_from.get(n, 0.0)) for n in goal.keys()}
        self._grip_goal  = {n: float(goal[n]) for n in goal.keys()}
        self._grip_ramp_T = float(T)
        self._grip_ramp_rate = float(rate_hz)

        # 기존 타이머 정리 후 새로 시작
        if self._grip_ramp_timer is not None:
            self._grip_ramp_timer.cancel()
            self._grip_ramp_timer = None

        self._grip_ramp_t0 = self.get_clock().now()
        period = 1.0 / max(1.0, self._grip_ramp_rate)
        self._grip_ramp_timer = self.create_timer(period, self._grip_ramp_tick)
        self.get_logger().info(f"Gripper ramp start: T={self._grip_ramp_T:.2f}s @ {self._grip_ramp_rate:.0f}Hz")





def main():
    rclpy.init()
    node = PathNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


                