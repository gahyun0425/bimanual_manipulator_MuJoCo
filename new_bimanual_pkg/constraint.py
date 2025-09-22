# constraint 환경 제약

import numpy as np
import rclpy
import math
import pinocchio as pin

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState, Constraints, PositionConstraint, OrientationConstraint, BoundingVolume
from moveit_msgs.srv import GetStateValidity
from geometry_msgs.msg import Pose, PoseStamped
from shape_msgs.msg import SolidPrimitive

# --------- 전역 노드/클라이언트 (여러 번 호출해도 1개만 유지) ---------
_NODE = None
_CLI = None

# move_group이 활성화 되었는지 확인
def _ensure_client(timeout: float = 2.0):
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
    global _NODE, _CLI
    if _NODE is not None:
        _NODE.destroy_node()
        _NODE = None
        _CLI = None
    if rclpy.ok():
        rclpy.shutdown()

def make_position_constraint(
    link_name: str,
    target_pose_world,   # PoseStamped
    eps_xyz=(0.08, 0.08, 0.07),  # 허용 반경(±m): x,y,z
) -> PositionConstraint:
    ex, ey, ez = eps_xyz

    prim = SolidPrimitive()
    prim.type = SolidPrimitive.BOX
    prim.dimensions = [2*ex, 2*ey, 2*ez]  # BOX 전체 길이 [X,Y,Z]

    bv = BoundingVolume()
    bv.primitives.append(prim)
    bv.primitive_poses.append(target_pose_world.pose)  # 박스 중심/자세 = 목표 포즈

    pc = PositionConstraint()
    pc.header.frame_id = target_pose_world.header.frame_id  # ← 목표 포즈 프레임 그대로
    pc.link_name = link_name
    pc.constraint_region = bv
    pc.target_point_offset.x = 0.0
    pc.target_point_offset.y = 0.0
    pc.target_point_offset.z = 0.0
    pc.weight = 1.0
    return pc



def make_loose_orientation_constraint(
    link_name: str,
    frame_id: str = "world",
    tol_xyz_deg: Tuple[float,float,float] = (180.0,180.0,180.0),
    quat: Tuple[float,float,float,float] = (0.0,0.0,0.0,1.0),
) -> OrientationConstraint:
    """자세 제약 (허용각 크게 주면 사실상 자유)"""
    oc = OrientationConstraint()
    oc.header.frame_id = frame_id
    oc.link_name = link_name
    oc.orientation.x, oc.orientation.y, oc.orientation.z, oc.orientation.w = quat
    oc.absolute_x_axis_tolerance = math.radians(tol_xyz_deg[0])
    oc.absolute_y_axis_tolerance = math.radians(tol_xyz_deg[1])
    oc.absolute_z_axis_tolerance = math.radians(tol_xyz_deg[2])
    oc.weight = 1.0
    return oc

def make_path_constraints(
    pcs: List[PositionConstraint],
    ocs: Optional[List[OrientationConstraint]] = None,
    name: str = "path_constraints",
) -> Constraints:
    cons = Constraints()
    cons.name = name
    cons.position_constraints = pcs
    if ocs:
        cons.orientation_constraints = ocs
    return cons

# --------- (옵션) 로테이션/플레이스 preset ---------
def build_rotation_constraints(ee_link: str, hold_pose_world: PoseStamped,
                               pos_sphere_mm: float = 15.0,
                               align_deg: float = 3.0,
                               frame_id: str = "world") -> Constraints:
    eps_m = (pos_sphere_mm/1000.0)   # mm -> m
    pc = make_position_constraint(ee_link, hold_pose_world,
                                  eps_xyz=(eps_m, eps_m, eps_m))  # ★ 수정
    oc = make_loose_orientation_constraint(ee_link, frame_id,
                                           tol_xyz_deg=(align_deg,180.0,180.0))
    return make_path_constraints([pc], [oc], name="rotate_phase")


def build_place_constraints(ee_link, place_pose_world,
                            eps_xy_mm=80.0, eps_z_mm=60.0,
                            level_deg=5.0):
    # 위치 박스 크기 (± -> BOX 길이 = 2*eps)
    eps = (eps_xy_mm/1000.0, eps_xy_mm/1000.0, eps_z_mm/1000.0)

    # PositionConstraint: 목표 포즈를 중심으로 한 BOX
    pc = make_position_constraint(ee_link, place_pose_world, eps_xyz=eps)

    # ★ OrientationConstraint: 목표 포즈의 쿼터니언을 그대로 사용
    ori = place_pose_world.pose.orientation
    oc = make_loose_orientation_constraint(
        ee_link,
        frame_id=place_pose_world.header.frame_id,
        tol_xyz_deg=(level_deg, level_deg, 10.0),
        quat=(ori.x, ori.y, ori.z, ori.w),
    )

    return make_path_constraints([pc], [oc], name="place_phase")



# --------- 충돌 + (옵션) Path Constraints 검사 ---------
def is_state_valid(   
        q: np.ndarray,
        joint_names: List[str],
        lb: np.ndarray,
        ub: np.ndarray,
        group_name: str = "manipulator",
        timeout: float = 2.0,
        constraints: Optional[Constraints] = None,
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

# --------- (NEW) 제약 투영 스펙 & 가우스-뉴턴 프로젝터 ---------
@dataclass
class ConstraintProjSpec:
    """
    다양체 제약:
      - function(q) -> h(q)  : (m,) 벡터 (0이면 만족)
      - jacobian(q)  -> J(q) : (m,N) or (N,m) 중 (m,N)로 맞춰 사용
    """
    function: Callable[[np.ndarray], np.ndarray]
    jacobian: Callable[[np.ndarray], np.ndarray]
    tol: float = 1e-1
    max_iters: int = 100
    damping: float = 1e-8  # 레벤버그-마콰르트용 람다

def project_q_gauss_newton(q_in: np.ndarray,
                           proj: ConstraintProjSpec,
                           lb: np.ndarray,
                           ub: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    임의의 q를 h(q)=0 다양체로 수치적으로 투영.
    반환: (q_proj, success)
    """
    q = np.asarray(q_in, float).copy()
    lb = np.asarray(lb, float); ub = np.asarray(ub, float)

    for _ in range(max(1, proj.max_iters)):
        h = np.atleast_1d(proj.function(q)).astype(float)   # (m,)
        if np.linalg.norm(h) <= proj.tol:
            return np.clip(q, lb, ub), True
        J = np.asarray(proj.jacobian(q), float)
        if J.ndim != 2:
            raise ValueError("jacobian(q) must be 2-D (m x N).")
        # (m x N)
        m, N = J.shape
        # 덤핑 포함 가우스-뉴턴 스텝: dq = - J^T (J J^T + λI)^(-1) h
        JJt = J @ J.T + proj.damping * np.eye(m)
        try:
            lam = np.linalg.solve(JJt, h)          # (m,)
        except np.linalg.LinAlgError:
            return q, False
        dq = - J.T @ lam                            # (N,)
        q = np.clip(q + dq, lb, ub)
    # 최종 검사
    return (np.clip(q, lb, ub), np.linalg.norm(proj.function(q)) <= proj.tol)

# --------- (NEW) MoveIt 유효성 콜백 빌더 ---------
def build_validity_cb(joint_names: List[str],
                      lb: np.ndarray,
                      ub: np.ndarray,
                      group_name: str,
                      constraints: Optional[Constraints],
                      timeout: float = 2.0) -> Callable[[np.ndarray], bool]:
    """RRT 등에 꽂을 수 있는 상태 유효성 콜백 생성"""
    def _valid(q: np.ndarray) -> bool:
        return is_state_valid(q, joint_names, lb, ub, group_name, timeout, constraints)
    return _valid

# --------- (NEW) 에지 유효성 검사: 선형보간 + (선택)투영 + 충돌/제약 체크 ---------
def check_motion_discrete(q_from: np.ndarray,
                          q_to: np.ndarray,
                          validity_cb: Callable[[np.ndarray], bool],
                          step: float = 0.05,
                          projector: Optional[ConstraintProjSpec] = None,
                          lb: Optional[np.ndarray] = None,
                          ub: Optional[np.ndarray] = None
                          ) -> Tuple[bool, List[np.ndarray]]:
    """
    선형 보간으로 [q_from -> q_to]를 쪼개며:
      - (선택) 각 보간점 투영(project_q_gauss_newton)
      - 유효성 검사(validity_cb)
    반환: (성공여부, 경로 샘플 리스트)
    """
    q_from = np.asarray(q_from, float); q_to = np.asarray(q_to, float)
    L = float(np.linalg.norm(q_to - q_from))
    if L <= 1e-9:
        return validity_cb(q_to), [q_from, q_to]

    n_seg = max(1, int(np.ceil(L / max(1e-6, step))))
    path = [q_from.copy()]
    q_prev = q_from.copy()

    for i in range(1, n_seg + 1):
        r = i / n_seg
        q_lin = (1.0 - r) * q_from + r * q_to  # 선형보간
        if projector is not None:
            if lb is None or ub is None:
                raise ValueError("projector 사용 시 lb/ub를 제공하세요.")
            q_proj, ok = project_q_gauss_newton(q_lin, projector, lb, ub)
            print("iter proj:", np.linalg.norm(projector.function(q_proj)), "ok?", ok)


            if not ok:
                return False, path
            q_chk = q_proj
        else:
            q_chk = q_lin

        if not validity_cb(q_chk):
            return False, path

        # 단조/후퇴 방지(선택): 필요하면 거리 비교 추가 가능
        path.append(q_chk)
        q_prev = q_chk

    return True, path

# --------- 단일 파일 테스트/예시 ---------
if __name__ == "__main__":
    try:
        # (예시) bimanual 그룹에서 왼/오른 EE 위치 고정(자세 자유)
        GROUP = "both_arms"
        LEFT_EE_LINK  = "left_ee_link"
        RIGHT_EE_LINK = "right_ee_link"
        FRAME = "world"  # MoveIt planning frame과 일치해야 함

        # 1) 고정할 EE 위치 (임시 값)
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

        pc_left  = make_position_constraint(LEFT_EE_LINK,  left_pose,
                                            eps_xyz=(0.003, 0.003, 0.003))   # ★ 수정
        pc_right = make_position_constraint(RIGHT_EE_LINK, right_pose,
                                            eps_xyz=(0.003, 0.003, 0.003))   # ★ 수정


        path_cons = make_path_constraints([pc_left, pc_right], name="bimanual_fix_ee_positions")

        # 2) 검사할 q / 조인트 정보 준비 (너의 로봇에 맞게 교체)
        joint_names = [
            "l_j1","l_j2","l_j3","l_j4","l_j5","l_j6","l_j7",
            "r_j1","r_j2","r_j3","r_j4","r_j5","r_j6","r_j7",
        ]
        N = len(joint_names)
        q0 = np.zeros(N)  # 테스트용
        q1 = np.zeros(N); q1[0] = 0.2  # 살짝 움직인 목표 (예시)
        lb = np.deg2rad(np.full(N, -170.0))
        ub = np.deg2rad(np.full(N,  170.0))

        # 3) 충돌+제약 검사 단일 상태 테스트
        ok = is_state_valid(
            q=q0,
            joint_names=joint_names,
            lb=lb, ub=ub,
            group_name=GROUP,
            timeout=2.0,
            constraints=path_cons,
        )
        print(f"[Check] valid with constraints? {ok}")

        # 4) (옵션) 제약 투영 스펙 (예: 단일 스칼라 제약 h(q)=a^T q - c = 0)
        #    실제 사용에선 FK/Jacobian으로 h(q), J(q)를 만들어 넣어라.
        a = np.zeros(N); a[0] = 1.0; c = 0.1  # 첫 관절을 0.1 rad로 유지하는 가짜 제약 예시
        proj_spec = ConstraintProjSpec(
            function=lambda q: np.array([a @ q - c]),
            jacobian=lambda q: np.array([a]),  # (1,N)
            tol=1e-4, max_iters=20, damping=1e-8
        )

        h0 = proj_spec.function(q0)
        J0 = proj_spec.jacobian(q0)

        print("h0 shape:", h0.shape, "norm:", np.linalg.norm(h0))
        print("J0 shape:", J0.shape)

        # 5) 경로 구간 유효성 검사 (선형보간 -> 투영 -> 충돌+제약 검사)
        validity_cb = build_validity_cb(joint_names, lb, ub, GROUP, path_cons, timeout=2.0)
        ok_edge, q_segment = check_motion_discrete(
            q_from=q0, q_to=q1,
            validity_cb=validity_cb,
            step=0.05,
            projector=proj_spec,   # 필요 없으면 None
            lb=lb, ub=ub
        )
        print(f"[EdgeCheck] ok={ok_edge}, samples={len(q_segment)}")

    finally:
        shutdown_client()
    
# pot 손잡이를 양팔로 잡았다는 rigid 제약 projector 생성 (수정본)
def build_pot_grasp_projector(model: pin.Model,
                              data: pin.Data,
                              ee_frame_l: str,
                              ee_frame_r: str,
                              T_pot0: pin.SE3,          # (미사용: 인터페이스 유지)
                              T_l_in_pot: pin.SE3,
                              T_r_in_pot: pin.SE3,
                              tol: float = 1e-1,
                              max_iters=50,
                              damping=1e-6
                              ) -> ConstraintProjSpec:

    fidL = model.getFrameId(ee_frame_l)
    fidR = model.getFrameId(ee_frame_r)

    def h(q: np.ndarray):
        # FK
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        T_wL = data.oMf[fidL]
        T_wR = data.oMf[fidR]

        # pot 추정 (왼손 기준), 오른손 예상 pose
        T_pot_est     = T_wL * T_l_in_pot.inverse()
        T_wR_expected = T_pot_est * T_r_in_pot

        # --- 위치 에러만 사용 ---
        err_pos = T_wR.translation - T_wR_expected.translation
        return 10.0 * err_pos    # (3,)

    def J(q: np.ndarray):
        # 모든 프레임 자코비안 계산
        pin.computeJointJacobians(model, data, q)
        pin.updateFramePlacements(model, data)

        T_wL = data.oMf[fidL]
        T_wR = data.oMf[fidR]
        T_pot_est     = T_wL * T_l_in_pot.inverse()
        T_wR_expected = T_pot_est * T_r_in_pot

        JL_w = pin.computeFrameJacobian(model, data, q, fidL, pin.WORLD)   # (6,N)
        JR_w = pin.computeFrameJacobian(model, data, q, fidR, pin.WORLD)   # (6,N)

        # 예상 오른손 자코비안 (왼팔 경유)
        Ad = (T_wR_expected * T_wL.inverse()).toActionMatrix()
        Jexp_from_L = Ad @ JL_w

        # 위치 row만 추출
        J_rigid_pos = JR_w[:3,:] - Jexp_from_L[:3,:]
        return J_rigid_pos   # (3,N)

    return ConstraintProjSpec(
        function=h,
        jacobian=J,
        tol=1.0,         # ★ 여기 tol 2cm 이상으로
        max_iters=2000,    # ★ 반복 늘려주고
        damping=1e-3,     # ★ 안정화용
    )
