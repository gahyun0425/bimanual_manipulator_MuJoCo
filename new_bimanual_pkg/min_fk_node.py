#!/usr/bin/env python3
import sys
import math

# ===== Robot DH-based geometric constants (meters) =====
d1 = 0.11     # base height
a2 = 0.1501   # link2 length
a3 = 0.145    # link3 length
d4 = 0.057    # end-effector offset

# ===== Joint angle offsets (radian) =====
theta_offset = [
    0.0,      # joint1
    1.301,    # joint2
    -1.301,   # joint3
    0.0       # joint4
]

def deg(rad): return rad * 180.0 / math.pi

def ik(x_cm, y_cm, z_cm):
    # cm -> m
    x, y, z = x_cm/100.0, y_cm/100.0, z_cm/100.0

    # ---- θ1 먼저 계산 ----
    theta1 = math.atan2(y, x)

    # wrist center 좌표 보정 (d4 만큼 x-y 평면에서 빼줌)
    xw = x - d4*math.cos(theta1)
    yw = y - d4*math.sin(theta1)
    zw = z

    # r, zrel
    r = math.sqrt(xw*xw + yw*yw)
    zrel = zw - d1
    s = math.sqrt(r*r + zrel*zrel)

    # law of cosines for θ3
    num = s*s - a2*a2 - a3*a3
    den = 2*a2*a3
    c3 = num/den
    c3 = max(-1.0, min(1.0, c3))
    s3_abs = math.sqrt(max(0.0, 1.0 - c3*c3))

    sols = []
    for s3_sign in (+1, -1):
        s3 = s3_sign * s3_abs
        t3 = math.atan2(s3, c3)
        t2 = math.atan2(zrel, r) - math.atan2(a3*s3, a2 + a3*c3)

        # joint4는 여기선 orientation 고정 → 0
        ths = [
            theta1 + theta_offset[0],
            t2     + theta_offset[1],
            t3     + theta_offset[2],
            0.0    + theta_offset[3]
        ]
        sols.append([deg(t) for t in ths])
    return sols

def main():
    print("목표 위치를 입력하세요 (x y z in cm): ", end="", flush=True)
    parts = sys.stdin.readline().strip().split()
    if len(parts) != 3:
        print("x y z 세 값을 입력해야 합니다.")
        sys.exit(1)
    x, y, z = map(float, parts)
    sols = ik(x, y, z)

    for i, sol in enumerate(sols, 1):
        print(f"Solution {i}: {sol[0]:.3f} {sol[1]:.3f} {sol[2]:.3f} {sol[3]:.3f}")

if __name__ == "__main__":
    main()
