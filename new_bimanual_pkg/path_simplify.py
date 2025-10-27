# new_bimanual_pkg/path_simplify.py
from __future__ import annotations
import numpy as np
import random
import time
from typing import Callable, Optional

# q와 constraint를 받아 유효/무효를 돌려주는 콜백의 타입 -> smoothing 과정에서 충돌과 리밋 검사 위해
IsValidFn = Callable[[np.ndarray, Optional[object]], bool]

# joint limit 확인
def _joint_limit_ok(q: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> bool:
    q  = np.asarray(q,  float)
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)
    if q.shape != lb.shape or lb.shape != ub.shape:
        return False
    if (q < lb).any() or (q > ub).any():
        return False
    return True

# 현재 q가 Moveit의 충돌/제약 검사를 통과하는지 서비스 호출로 확인
# q와 constraint가 유효하면 True 반환
def _call_moveit_validity(node,
                          q: np.ndarray,
                          constraints: Optional[object] = None,
                          timeout: float = 0.25) -> bool:
    
    required = ("joint_names", "lb", "ub", "group_name", "state_valid_cli")
    if not all(hasattr(node, k) for k in required):
        return False

    if not _joint_limit_ok(q, node.lb, node.ub):
        return False

    cli = getattr(node, "state_valid_cli", None)
    if cli is None:
        return False

    try:
        from sensor_msgs.msg import JointState
        from moveit_msgs.msg import RobotState
        from moveit_msgs.srv import GetStateValidity

        js = JointState(name=list(node.joint_names),
                        position=np.asarray(q, float).tolist())
        rs = RobotState(joint_state=js)

        req = GetStateValidity.Request()
        req.robot_state = rs
        req.group_name = str(node.group_name)
        if constraints is not None:
            req.constraints = constraints

        fut = cli.call_async(req)

        # 1) 우선 spin_until_future_complete 시도
        try:
            import rclpy
            rclpy.spin_until_future_complete(node, fut, timeout_sec=float(timeout))
        except Exception:
            # 2) 실패 시 폴링으로 폴백
            t0 = time.time()
            while not fut.done():
                if time.time() - t0 > timeout:
                    try:
                        node.get_logger().warn("path_simplify: validity timeout")
                    except Exception:
                        pass
                    return False
                time.sleep(0.001)

        if not fut.done():
            return False

        res = fut.result()
        return bool(res and res.valid)

    except Exception:
        return False

# 위의 Moveit 검사를 콜백 형태로 만들어 돌려 줌. -> is valid처럼 간단히 부를 수 있는 함수 생성
def make_is_valid_from_node(node):
    def _cb(q: np.ndarray, constraints: Optional[object] = None) -> bool:
        return _call_moveit_validity(node, q, constraints, timeout=0.25)
    return _cb

# 직선 보간을 통해 중간 점들이 전부 유효함지 체크 -> 중간 샘플 다 통과하면 True 반환
def _edge_is_valid(q1: np.ndarray,
                   q2: np.ndarray,
                   is_valid: IsValidFn,
                   constraints: Optional[object] = None,
                   max_step: float = 0.012,
                   min_samples: int = 3) -> bool:

    q1 = np.asarray(q1, float); q2 = np.asarray(q2, float)
    seg_len = float(np.linalg.norm(q2 - q1)) + 1e-12
    n = max(min_samples, int(np.ceil(seg_len / max(1e-9, max_step))))
    # 내부 샘플만 검사 (끝점은 이미 유효하다고 가정)
    for k in range(1, n):  # 1..n-1
        a = k / n
        q = (1.0 - a) * q1 + a * q2
        if not is_valid(q, constraints):
            return False
    return True

# 경로 간소화 과정 -> 중간 포인트가 필요 없으면 지움 -> 포인트 수 감소 (불필요한 굴곡 제거) -> 후 처리 안정/고속
def reduce_vertices(path: np.ndarray,
                    is_valid: IsValidFn,
                    constraints: Optional[object] = None,
                    max_step: float = 0.012,
                    passes: int = 3) -> np.ndarray:

    if path is None or len(path) < 3:
        return np.asarray(path, float)
    P = np.asarray(path, float)
    for _ in range(max(1, passes)):
        keep = [0]
        i = 0
        while i < len(P) - 2:
            if _edge_is_valid(P[i], P[i+2], is_valid, constraints, max_step=max_step):
                j = i + 2
                while j + 1 < len(P) and _edge_is_valid(P[i], P[j+1], is_valid, constraints, max_step=max_step):
                    j += 1
                keep.append(j)
                i = j
            else:
                keep.append(i + 1)
                i += 1
        if keep[-1] != len(P) - 1:
            keep.append(len(P) - 1)

        # 순서 보존 + 중복 제거
        new_keep = []
        seen = set()
        for idx in keep:
            if idx not in seen:
                new_keep.append(idx)
                seen.add(idx)
        P = P[np.asarray(new_keep, dtype=int)]
    return P


# 비용(길이 + 곡률) 경로의 길이 + 곡률(라플라시안 제곱)을 합친 점수 계산 =? 쇼트컷을 할지 말지 정량적으로 판단하기 위해
def _path_cost(P: np.ndarray, lam_len: float = 1.0, lam_curv: float = 4.0) -> float:
    P = np.asarray(P, float)
    if len(P) < 2:
        return 0.0
    L = float(np.sum(np.linalg.norm(P[1:] - P[:-1], axis=1)))
    if len(P) < 3:
        return lam_len * L
    lap = P[:-2] - 2.0 * P[1:-1] + P[2:]
    C = float(np.sum(np.sum(lap * lap, axis=1)))
    return lam_len * L + lam_curv * C

# 쇼트컷(곡률인지) -> 경로의 임의 구간이 직선으로 대체 가능한지 확인하고, 충돌 ok + 비용감소 할 때만 수락
# 충돌은 지키면서 지그재그/뾰족 코너가 눈에 띄게 줄어듦
def shortcut_path_curvaware(path: np.ndarray,
                            is_valid: IsValidFn,
                            constraints: Optional[object] = None,
                            max_step: float = 0.012,
                            attempts: int = 900,
                            lam_len: float = 1.0,
                            lam_curv: float = 4.0,
                            rng: Optional[random.Random] = None) -> np.ndarray:

    if path is None or len(path) < 3:
        return np.asarray(path, float)
    P = np.asarray(path, float).copy()
    rnd = rng or random
    base_cost = _path_cost(P, lam_len, lam_curv)

    for _ in range(max(1, attempts)):
        if len(P) < 3:
            break
        i = rnd.randint(0, len(P) - 3)
        j = rnd.randint(i + 2, len(P) - 1)
        if _edge_is_valid(P[i], P[j], is_valid, constraints,
                          max_step=max_step, min_samples=3):
            cand = np.vstack([P[:i+1], P[j:]])
            new_cost = _path_cost(cand, lam_len, lam_curv)
            if new_cost + 1e-12 < base_cost:
                P = cand
                base_cost = new_cost
    return P

# 각 중간 점을 이웃과 평균에 가깝게 조금씩 이동해 모서리를 둥글게 바꾼 뒤 양옆 선분이 충돌 없이 유효할 때만 반영
def laplacian_smooth(path: np.ndarray,
                     is_valid: IsValidFn,
                     constraints: Optional[object] = None,
                     step: float = 0.35,
                     iters: int = 25,
                     max_step: float = 0.012) -> np.ndarray:

    if path is None or len(path) < 3:
        return np.asarray(path, float)
    P = np.asarray(path, float).copy()
    N = len(P)
    for _ in range(max(1, iters)):
        changed = False
        for i in range(1, N - 1):
            p_old = P[i].copy()
            p_new = p_old + step * (P[i-1] - 2.0 * p_old + P[i+1])
            ok = (_edge_is_valid(P[i-1], p_new, is_valid, constraints,
                                 max_step=max_step, min_samples=3)
                  and
                  _edge_is_valid(p_new, P[i+1], is_valid, constraints,
                                 max_step=max_step, min_samples=3))
            if ok:
                P[i] = p_new
                changed = True
        if not changed:
            break
    return P

# 밀도 보강 -> 이웃 포인트 사이 간격이 max_step 넘지 않도록 중간 점 추가
def densify_by_maxstep(path: np.ndarray, max_step: float = 0.012) -> np.ndarray:

    if path is None or len(path) < 2:
        return np.asarray(path, float)
    P = np.asarray(path, float)
    out = [P[0]]
    for i in range(len(P) - 1):
        a, b = P[i], P[i+1]
        seg = float(np.linalg.norm(b - a))
        n = max(1, int(np.ceil(seg / max(1e-9, max_step))))
        for k in range(1, n + 1):
            out.append((1 - k/n) * a + (k/n) * b)
    return np.asarray(out, float)

# 전체 파이프라인
def simplify_path(path: np.ndarray,
                  is_valid: IsValidFn,
                  constraints: Optional[object] = None,
                  *,
                  max_step: float = 0.012,
                  red_passes: int = 3,
                  shortcut_attempts: int = 900,
                  lam_len: float = 1.0,
                  lam_curv: float = 4.0,
                  smooth_iters: int = 25,
                  smooth_step: float = 0.35,
                  do_densify: bool = False,
                  rng: Optional[random.Random] = None) -> np.ndarray:
    
    if path is None or len(path) < 2:
        return np.asarray(path, float)

    P = np.asarray(path, float)
    P = reduce_vertices(P, is_valid, constraints, max_step=max_step, passes=red_passes)
    P = shortcut_path_curvaware(P, is_valid, constraints, max_step=max_step,
                                attempts=shortcut_attempts,
                                lam_len=lam_len, lam_curv=lam_curv, rng=rng)
    P = laplacian_smooth(P, is_valid, constraints, step=smooth_step,
                         iters=smooth_iters, max_step=max_step)
    if do_densify:
        P = densify_by_maxstep(P, max_step=max_step)
    return P
