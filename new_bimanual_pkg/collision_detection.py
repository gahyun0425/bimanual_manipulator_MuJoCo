# joint limit check & collision detection
# ee 고정 전

import numpy as np  # C++ eigen
import rclpy # ROS2 Python Client

from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState # Moveit robot state
from moveit_msgs.srv import GetStateValidity # 충돌 여부

# 노드 초기화 단계 -> 함수가 여러번 호출돼도 node/client를 매번 만들지 않기 위해
_NODE = None 
_CLI = None 

# move_group 실행 확인
def _ensure_client(timeout: float = 2.0):
    global _NODE, _CLI
    if not rclpy.ok():                 # 아직 init 안 됐으면
        rclpy.init(args=None)
    if _NODE is None:
        _NODE = rclpy.create_node("state_validity_client")
    if _CLI is None:
        _CLI = _NODE.create_client(GetStateValidity, "/check_state_validity")
        if not _CLI.wait_for_service(timeout_sec=timeout):
            _NODE.get_logger().error("/check_state_validity 서비스가 없습니다. move_group를 먼저 실행하세요.")
            raise RuntimeError("GetStateValidity not available")
        
# joint limit 판단. Moveit planning scene에서 충돌 판단 후 반환
def is_state_valid(
        q: np.ndarray, # 검사할 관절각 벡터
        joint_names: list, # 관절 이름 리스트 (Moveit/urdf 순서와 동일해야 함.)
        lb: np.ndarray, # min limit
        ub: np.ndarray, # max limit
        group_name: str = "manipulator", # Moveit planning 그룹명
        timeout: float = 2.0, # 서비스 준비/ 응답 대기 시간(초)
) -> bool:
    
    # 입력을 numpy array로 변환
    q = np.asarray(q, dtype=float)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    # joint limit check
    # 배열 길이 확인
    if len(joint_names) != len(q) or len(q) != len(lb) or len(lb) != len(ub):
        raise ValueError("joint_names, q, lb, ub 길이가 일치해야 합니다.")
    
    if(q < lb).any() or (q > ub).any(): # 한계 밖이면 유효하지 않음
        return False
    
    _ensure_client(timeout=timeout)

    # 요청 메시지 구성
    js = JointState(name=list(joint_names), position=q.tolist()) # 관절 이름/각도
    rs = RobotState(joint_state=js) # 로봇 상태 래핑
    req = GetStateValidity.Request(robot_state=rs, group_name=group_name) # 추가 제약 없으면 빈 객체


    # 호출 및 대기
    future = _CLI.call_async(req)
    rclpy.spin_until_future_complete(_NODE, future, timeout_sec=timeout) # 이벤트 루프 돌려 응답 대기
    if not future.done() or future.result() is None: # 타임아웃 / 실패 처리
        _NODE.get_logger().warn("GetStateBalidity timeout/fail")
        return False
    
    # 결과 반환
    return bool(future.result().valid)

def shutdown_client():
    global _NODE, _CLI
    if _NODE is not None:
        _NODE.destroy_node()
        _NODE = None
        _CLI = None
    if rclpy.ok():
        rclpy.shutdown()
        