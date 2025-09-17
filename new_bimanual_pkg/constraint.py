# check_validity_with_constraints.py
# - MoveIt의 /check_state_validity 서비스로
#   1) 조인트 리밋
#   2) 충돌
#   3) (옵션) Path Constraints
#   를 한 번에 검사하는 단일 파일 스크립트

import numpy as np
import rclpy

from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState, Constraints, PositionConstraint, OrientationConstraint, BoundingVolume
from moveit_msgs.srv import GetStateValidity
from geometry_msgs.msg import PoseStamped
from shape_msgs.msg import SolidPrimitive

# --------- 전역 노드/클라이언트 (여러 번 호출해도 1개만 유지) ---------
_NODE = None
_CLI = None

def _ensure_client(timeout: float = 2.0):
    """move_group가 띄워져 있고 /check_state_validity 서비스가 준비됐는지 확인"""
    global _NODE, _CLI
    if not rclpy.ok():
        rclpy.init(args=None)
    if _NODE is None:
        _NODE = rclpy.create_node("state_validity_client")
    if _CLI is None:
        _CLI = _NODE.create_client(GetStateValidity, "/check_state_validity")
        if not _CLI.wait_for_service(timeout_sec=timeout):
            _NODE.get_logger().error("/check_state_validity 서비스가 없습니다. move_group를 먼저 실행하세요.")
            raise RuntimeError("GetStateValidity not available")

def shutdown_client():
    """노드/클라이언트 정리"""
    global _NODE, _CLI
    if _NODE is not None:
        _NODE.destroy_node()
        _NODE = None
        _CLI = None
    if rclpy.ok():
        rclpy.shutdown()

# --------- 제약(Constraints) 생성 헬퍼들 ---------
def make_position_constraint(
    link_name: str,
    target_pose_world: PoseStamped,
    eps: float = 0.003,  # 허용 박스 크기 (3mm 권장 시작값)
) -> PositionConstraint:
    """EE 위치를 작은 박스 안에 묶는 PositionConstraint 생성"""
    prim = SolidPrimitive()
    prim.type = SolidPrimitive.BOX
    prim.dimensions = [eps, eps, eps]

    bv = BoundingVolume()
    bv.primitives.append(prim)
    bv.primitive_poses.append(target_pose_world.pose)

    pc = PositionConstraint()
    pc.header = target_pose_world.header        # frame_id는 보통 "world" 또는 planning_frame
    pc.link_name = link_name
    pc.target_point_offset.x = 0.0              # 링크 원점 고정
    pc.target_point_offset.y = 0.0
    pc.target_point_offset.z = 0.0
    pc.constraint_region = bv
    pc.weight = 1.0
    return pc

def make_loose_orientation_constraint(
    link_name: str,
    frame_id: str = "world",
) -> OrientationConstraint:
    """사실상 자세를 자유롭게 두고 싶을 때 아주 큰 허용오차로 만드는 OC"""
    oc = OrientationConstraint()
    oc.header.frame_id = frame_id
    oc.link_name = link_name
    oc.orientation.w = 1.0
    oc.absolute_x_axis_tolerance = 3.14159
    oc.absolute_y_axis_tolerance = 3.14159
    oc.absolute_z_axis_tolerance = 3.14159
    oc.weight = 1.0
    return oc

def make_path_constraints(
    pcs: list[PositionConstraint],
    ocs: list[OrientationConstraint] | None = None,
    name: str = "path_constraints",
) -> Constraints:
    cons = Constraints()
    cons.name = name
    cons.position_constraints = pcs
    if ocs:
        cons.orientation_constraints = ocs
    return cons

# 유효성 검사
def is_state_valid(   
        q: np.ndarray,
        joint_names: list,
        lb: np.ndarray,
        ub: np.ndarray,
        group_name: str = "manipulator",
        timeout: float = 2.0,
        constraints: Constraints | None = None,   # ★ 추가: 제약 함께 검사
) -> bool:
    """주어진 q가 조인트 리밋/충돌/제약을 만족하는지 검사"""
    q = np.asarray(q, dtype=float); lb = np.asarray(lb, dtype=float); ub = np.asarray(ub, dtype=float)

    # 길이/리밋 체크
    if len(joint_names) != len(q) or len(q) != len(lb) or len(lb) != len(ub):
        raise ValueError("joint_names, q, lb, ub 길이가 일치해야 합니다.")
    if (q < lb).any() or (q > ub).any():
        return False

    _ensure_client(timeout=timeout)

    # 요청 구성
    js = JointState(name=list(joint_names), position=q.tolist())
    rs = RobotState(joint_state=js)
    req = GetStateValidity.Request(robot_state=rs, group_name=group_name)

    if constraints is not None:
        req.constraints = constraints

    # 호출 및 응답 대기
    future = _CLI.call_async(req)
    rclpy.spin_until_future_complete(_NODE, future, timeout_sec=timeout)
    if not future.done() or future.result() is None:
        _NODE.get_logger().warn("GetStateValidity timeout/fail")
        return False

    return bool(future.result().valid)

# --------- 단일 파일 테스트/예시 ---------
if __name__ == "__main__":
    try:
        # (예시) bimanual 그룹에서 왼/오른 EE 위치 고정(자세 자유)
        # 1) 링크/그룹/프레임 이름은 네 SRDF/MoveIt 설정에 맞춰 수정
        GROUP = "both_arms"
        LEFT_EE_LINK  = "left_ee_link"
        RIGHT_EE_LINK = "right_ee_link"
        FRAME = "world"  # MoveIt planning frame과 일치해야 함

        # 2) 고정할 EE 위치 (현재 측정/추정값으로 채워라)
        left_pose = PoseStamped()
        left_pose.header.frame_id = FRAME
        left_pose.pose.position.x = 0.50
        left_pose.pose.position.y = 0.30
        left_pose.pose.position.z = 1.00
        left_pose.pose.orientation.w = 1.0

        right_pose = PoseStamped()
        right_pose.header.frame_id = FRAME
        right_pose.pose.position.x = 0.50
        right_pose.pose.position.y = -0.30
        right_pose.pose.position.z = 1.00
        right_pose.pose.orientation.w = 1.0

        pc_left  = make_position_constraint(LEFT_EE_LINK,  left_pose,  eps=0.003)
        pc_right = make_position_constraint(RIGHT_EE_LINK, right_pose, eps=0.003)

        # 자세는 자유로 둘 거라면 OrientationConstraint는 생략 가능
        # oc_left  = make_loose_orientation_constraint(LEFT_EE_LINK, FRAME)
        # oc_right = make_loose_orientation_constraint(RIGHT_EE_LINK, FRAME)

        path_cons = make_path_constraints([pc_left, pc_right], name="bimanual_fix_ee_positions")

        # 3) 검사할 q / 조인트 정보 준비 (너의 로봇에 맞게 교체)
        joint_names = [
            # 예시: 왼팔 7 + 오른팔 7
            "l_j1","l_j2","l_j3","l_j4","l_j5","l_j6","l_j7",
            "r_j1","r_j2","r_j3","r_j4","r_j5","r_j6","r_j7",
        ]
        N = len(joint_names)
        q  = np.zeros(N)  # 테스트용 (실제 값으로 대체)
        lb = np.deg2rad(np.full(N, -170.0))
        ub = np.deg2rad(np.full(N,  170.0))

        ok = is_state_valid(
            q=q,
            joint_names=joint_names,
            lb=lb, ub=ub,
            group_name=GROUP,
            timeout=2.0,
            constraints=path_cons,  # ★ 제약 같이 보냄
        )
        print(f"[Check] valid with constraints? {ok}")

    finally:
        shutdown_client()
