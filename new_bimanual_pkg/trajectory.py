# topp-ra & cubic spline interpolation

from __future__ import annotations
import math
from typing import Tuple, Dict
import numpy as np

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration


EPS = 1e-12


# -----------------------------
# 시간 기반 Cubic Hermite 스플라인 (다차원)
# -----------------------------
class TimeHermiteSplineND:
    def __init__(self, t_knots: np.ndarray, Q: np.ndarray, V: np.ndarray):
        """
        t_knots: (N,) 시간 매듭
        Q: (N,D) 각 매듭에서의 위치(조인트 값)
        V: (N,D) 각 매듭에서의 속도(조인트 속도)
        """
        self.t = np.asarray(t_knots, float)
        self.Q = np.asarray(Q, float)
        self.V = np.asarray(V, float)
        assert self.t.ndim == 1 and self.Q.ndim == 2 and self.V.shape == self.Q.shape
        self.N, self.D = self.Q.shape
        if self.N < 2:
            raise ValueError("스플라인 매듭은 최소 2개 이상 필요")

    def _seg_idx(self, tt: float) -> int:
        # 주어진 시간 tt가 포함되는 구간 인덱스 k 반환 (t[k] <= tt <= t[k+1])
        if tt <= self.t[0]:
            return 0
        if tt >= self.t[-1]:
            return self.N - 2
        return int(np.searchsorted(self.t, tt) - 1)

    @staticmethod
    def _basis(u: float):
        # Hermite basis 값
        u2 = u * u
        u3 = u2 * u
        h00 =  2*u3 - 3*u2 + 1
        h10 =      u3 - 2*u2 + u
        h01 = -2*u3 + 3*u2
        h11 =      u3 -   u2
        return h00, h10, h01, h11

    @staticmethod
    def _basis_d(u: float):
        # Hermite basis 도함수
        h00p =  6*u*u - 6*u
        h10p =  3*u*u - 4*u + 1
        h01p = -6*u*u + 6*u
        h11p =  3*u*u - 2*u
        return h00p, h10p, h01p, h11p

    def eval(self, tt: float) -> np.ndarray:
        # q(t) 위치
        k = self._seg_idx(tt)
        t0, t1 = self.t[k], self.t[k+1]
        h = max(t1 - t0, EPS)
        u = (tt - t0) / h
        h00, h10, h01, h11 = self._basis(u)
        q0, v0 = self.Q[k], self.V[k]
        q1, v1 = self.Q[k+1], self.V[k+1]
        return h00*q0 + h10*h*v0 + h01*q1 + h11*h*v1

    def eval_d(self, tt: float) -> np.ndarray:
        # q̇(t) 속도
        k = self._seg_idx(tt)
        t0, t1 = self.t[k], self.t[k+1]
        h = max(t1 - t0, EPS)
        u = (tt - t0) / h
        h00p, h10p, h01p, h11p = self._basis_d(u)
        q0, v0 = self.Q[k], self.V[k]
        q1, v1 = self.Q[k+1], self.V[k+1]
        dq_du = h00p*q0 + h10p*h*v0 + h01p*q1 + h11p*h*v1
        return dq_du / h


# -----------------------------
# 선형 경로 TOPP (끝 윈도우에서만 0으로 수렴)
# -----------------------------
def topp_over_path(
    Q: np.ndarray,
    qd_max: np.ndarray,
    qdd_max: np.ndarray,
    ds: float = 1e-3,
    sdot_start: float = 0.0,
    stop_window_s: float = 0.05,
    alpha_floor: float = 0.2,
    v_min_time: float = 1e-4,
) -> Dict[str, np.ndarray]:
    """
    입력:
      Q: (N,D) 조인트 웨이포인트
      qd_max: (D,) 조인트별 속도상한
      qdd_max: (D,) 조인트별 가속도상한
      ds: s-도메인 샘플 간격
      sdot_start: 시작 ṡ
      stop_window_s: s∈[1-stop_window_s,1]에서 0으로 감속
      alpha_floor: 중앙부에서 v >= alpha * vmax(s) 보장(0으로 두면 비활성)
      v_min_time: 시간 적분 시 평균속도 하한 (총시간 폭주 방지)
    출력(dict):
      s_grid, v(s), t_grid, seg_id, s_knots, t_knots, qdot_knots
    """
    Q = np.asarray(Q, float)
    qd_max = np.asarray(qd_max, float)
    qdd_max = np.asarray(qdd_max, float)
    N, D = Q.shape
    if N < 2:
        raise ValueError("웨이포인트는 최소 2개 이상 필요")

    # s-knots (거리 정규화)
    seg_len = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    s = np.zeros(N); s[1:] = np.cumsum(seg_len)
    L = s[-1]
    if L <= EPS:
        # 모든 점이 같다 → 시간 0, 속도 0
        t_knots = np.zeros(N)
        qdot_knots = np.zeros_like(Q)
        return dict(
            s_grid=np.array([0.0, 1.0]),
            v=np.zeros(2),
            t_grid=t_knots.copy(),
            seg_id=np.array([0, 0]),
            s_knots=np.linspace(0, 1, N),
            t_knots=t_knots,
            qdot_knots=qdot_knots,
        )

    s_knots = s / L
    seg_s = np.maximum(np.diff(s_knots), EPS)
    dqds = [(Q[k+1] - Q[k]) / seg_s[k] for k in range(N-1)]

    def speed_cap(dqds_k: np.ndarray) -> float:
        # |dq/ds| * ṡ <= q̇_max → ṡ <= min_i qd_max[i] / |dqds[i]|
        caps = [qd_max[i] / abs(dqds_k[i]) for i in range(D) if abs(dqds_k[i]) > 1e-12]
        return min(caps) if caps else 1e6

    def acc_bounds(dqds_k: np.ndarray) -> Tuple[float, float]:
        # |dq/ds * s̈| <= q̈_max → s̈ ∈ [-min_i(qdd_max/|dqds|), +min_i(...)]
        lo, hi = -1e9, 1e9
        for i in range(D):
            a = abs(dqds_k[i])
            if a <= 1e-12:
                continue
            b = qdd_max[i] / a
            lo = max(lo, -b)
            hi = min(hi, b)
        return lo, hi

    # s-grid
    s_grid = [s_knots[0]]; seg_id = [0]
    for k in range(N-1):
        s0, s1 = s_knots[k], s_knots[k+1]
        n = max(1, int(math.ceil((s1 - s0) / ds)))
        for j in range(1, n+1):
            s_grid.append(min(s0 + j*(s1 - s0)/n, s1))
            seg_id.append(k)
    s_grid = np.array(s_grid, float)
    seg_id = np.array(seg_id, int)
    M = len(s_grid)

    # forward pass
    v = np.zeros(M, float); v[0] = float(sdot_start)
    for k in range(M-1):
        dqds_k = dqds[seg_id[k]]
        vmax = speed_cap(dqds_k)
        v[k] = min(v[k], vmax)
        lo, hi = acc_bounds(dqds_k)
        a = max(0.0, hi)
        ds_k = s_grid[k+1] - s_grid[k]
        v[k+1] = math.sqrt(max(0.0, v[k]*v[k] + 2*a*ds_k))

    # backward pass
    for k in range(M-2, -1, -1):
        dqds_k = dqds[seg_id[k+1]]
        vmax = speed_cap(dqds_k)
        v[k+1] = min(v[k+1], vmax)
        lo, hi = acc_bounds(dqds_k)
        a = min(0.0, lo)
        ds_k = s_grid[k+1] - s_grid[k]
        v_prev = math.sqrt(max(0.0, v[k+1]*v[k+1] - 2*abs(a)*ds_k))
        v[k] = min(v[k], v_prev)

    # 중앙부 cap-비율 바닥속도(선택)
    if alpha_floor > 0.0:
        vmax_grid = np.array([speed_cap(dqds[seg_id[i]]) for i in range(M)], float)
        sw = max(0.0, min(1.0, stop_window_s))
        s_start = max(0.0, 1.0 - sw) if sw > 0.0 else 1.0
        for k in range(M-1):  # 마지막은 0으로 갈 수 있으니 제외
            if s_grid[k] < s_start:
                v[k] = min(max(v[k], alpha_floor * vmax_grid[k]), vmax_grid[k])

    # 끝 윈도우에서만 0으로 수렴
    sw = max(0.0, min(1.0, stop_window_s))
    if sw > 0.0:
        s_start = max(0.0, 1.0 - sw)
        vmax_grid = np.array([speed_cap(dqds[seg_id[i]]) for i in range(M)], float)
        idx_start = int(np.searchsorted(s_grid, s_start))
        idx_start = min(max(idx_start, 0), M-2)
        v_ref = max(v[idx_start], 1e-9)
        for k in range(idx_start, M):
            s_cur = s_grid[k]
            # raised-cosine taper: s_start→1.0 에서 1→0
            tau = 0.5 * (1.0 + math.cos(math.pi * (s_cur - s_start) / max(sw, EPS))) if s_cur >= s_start else 1.0
            tau = max(0.0, min(1.0, tau))
            v[k] = min(v[k], tau * v_ref, vmax_grid[k])
        v[-1] = 0.0

    # 시간 적분
    t_grid = np.zeros_like(s_grid)
    V_MIN_TIME = float(max(v_min_time, 1e-8))
    for k in range(1, M):
        ds_k = s_grid[k] - s_grid[k-1]
        vavg = 0.5*(v[k] + v[k-1])
        if not np.isfinite(vavg):
            vavg = 0.0
        vavg = max(vavg, V_MIN_TIME)
        t_grid[k] = t_grid[k-1] + ds_k / vavg

    # 웨이포인트 시각/속도
    sdot_knots = np.interp(s_knots, s_grid, v)
    qdot_knots = np.zeros_like(Q)
    for k in range(N-1):
        qdot_knots[k] = dqds[k] * sdot_knots[k]
    qdot_knots[-1] = np.zeros(D)  # 종단 속도 0
    t_knots = np.interp(s_knots, s_grid, t_grid)

    return dict(
        s_grid=s_grid, v=v, t_grid=t_grid, seg_id=seg_id,
        s_knots=s_knots, t_knots=t_knots, qdot_knots=qdot_knots
    )


# -----------------------------
# 스플라인 구성 + 균일 시간 샘플링
# -----------------------------
def build_spline(t_knots: np.ndarray, Q: np.ndarray, V: np.ndarray) -> TimeHermiteSplineND:
    """(t_k, Q_k, V_k)로 다차원 Hermite 스플라인 객체 생성."""
    return TimeHermiteSplineND(t_knots, Q, V)


def sample_spline_uniform(
    spline: TimeHermiteSplineND,
    sample_hz: float = 50.0,
    max_points: int = 100_000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    스플라인을 균일 시간 간격으로 샘플링.
    반환: (t_samples(M,), Q_samples(M,D), Qd_samples(M,D))
    """
    t0, t1 = float(spline.t[0]), float(spline.t[-1])
    T_total = t1 - t0
    if (not np.isfinite(T_total)) or (T_total <= 0.0):
        # 비정상: 매듭만 반환
        t_samples = spline.t.copy()
        Qs = np.vstack([spline.eval(tt) for tt in t_samples])
        Qds = np.vstack([spline.eval_d(tt) for tt in t_samples])
        return t_samples, Qs, Qds

    M = int(max(2, math.ceil(T_total * float(sample_hz))))
    if M > int(max_points):
        M = int(max_points)
    t_samples = np.linspace(t0, t1, M)
    Qs = np.vstack([spline.eval(tt) for tt in t_samples])
    Qds = np.vstack([spline.eval_d(tt) for tt in t_samples])
    return t_samples, Qs, Qds

def clamp_eval(spline, t):
    # 스플라인의 정의역 [t_knots[0], t_knots[-1]] 밖이면 경계값으로 평가
    t0, t1 = float(spline.t[0]), float(spline.t[-1])
    if t <= t0:
        q  = spline.eval(t0)
        qd = np.zeros_like(q)  # 시작속도도 원하면 0으로
        return q, qd
    if t >= t1:
        q  = spline.eval(t1)
        qd = np.zeros_like(q)  # 끝에서는 반드시 0
        return q, qd
    return spline.eval(t), spline.eval_d(t)

# -----------------------------
# 원샷 파이프라인 (편의)
# -----------------------------
def plan_trajectory(
    Q: np.ndarray,
    qd_max: np.ndarray,
    qdd_max: np.ndarray,
    *,
    ds: float = 1e-3,
    sdot_start: float = 0.0,
    stop_window_s: float = 0.05,
    alpha_floor: float = 0.2,
    v_min_time: float = 1e-4,
    sample_hz: float = 50.0,
    max_points: int = 100_000,
) -> Dict[str, np.ndarray]:
    topp = topp_over_path(
        Q, qd_max, qdd_max, ds=ds, sdot_start=sdot_start,
        stop_window_s=stop_window_s, alpha_floor=alpha_floor, v_min_time=v_min_time
    )
    spline = build_spline(topp["t_knots"], Q, topp["qdot_knots"])
    t_samples, Q_samples, Qd_samples = sample_spline_uniform(
        spline, sample_hz=sample_hz, max_points=max_points
    )
    return dict(
        t_knots=topp["t_knots"],
        qdot_knots=topp["qdot_knots"],
        t_samples=t_samples,
        Q_samples=Q_samples,
        Qd_samples=Qd_samples,
        s_grid=topp["s_grid"], v=topp["v"], t_grid=topp["t_grid"]
    )

def make_joint_trajectory_msg(joint_names, t0, t_samples, Q_samples, Qd_samples=None, hold_sec=1.0):
    traj = JointTrajectory()
    traj.joint_names = list(joint_names)

    # 샘플 기록
    for i, tt in enumerate(t_samples):
        pt = JointTrajectoryPoint()
        pt.positions = Q_samples[i].tolist()
        if Qd_samples is not None:
            pt.velocities = Qd_samples[i].tolist()
        tau = max(0.0, float(tt - t0))
        pt.time_from_start = Duration(sec=int(tau), nanosec=int((tau - int(tau)) * 1e9))
        traj.points.append(pt)

    # 1) 마지막 속도 0으로 강제
    if len(traj.points) > 0:
        traj.points[-1].velocities = [0.0] * len(joint_names)

        # 2) 마지막 포지션/속도 0으로 한 번 더 추가(홀드)
        last = traj.points[-1]
        hold = JointTrajectoryPoint()
        hold.positions  = list(last.positions)
        hold.velocities = [0.0] * len(joint_names)
        tau = (last.time_from_start.sec + last.time_from_start.nanosec * 1e-9) + float(hold_sec)
        hold.time_from_start = Duration(sec=int(tau), nanosec=int((tau - int(tau)) * 1e9))
        traj.points.append(hold)

    return traj
