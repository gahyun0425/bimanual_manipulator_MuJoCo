#forward kinematics

import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import numpy as np

# 1) URDF 로드
urdf_path = '/home/gaga/bimanual_ws/src/new_bimanual_pkg/mujoco_models/bimanual.urdf'
robot = RobotWrapper.BuildFromURDF(urdf_path, package_dirs=[])
model = robot.model
data  = model.createData()

# 2) EE 프레임
ee_frame_l = 'gripper_l_rh_p12_rn_r2'
ee_frame_r = 'gripper_r_rh_p12_rn_r2'
fid_l = model.getFrameId(ee_frame_l)
fid_r = model.getFrameId(ee_frame_r)

# 3) 조인트 인덱스 준비
left_joint_names  = [f'arm_l_joint{i}' for i in range(1, 8)]
right_joint_names = [f'arm_r_joint{i}' for i in range(1, 8)]

left_joint_ids  = [model.getJointId(nm) for nm in left_joint_names]
right_joint_ids = [model.getJointId(nm) for nm in right_joint_names]

assert all(jid > 0 for jid in left_joint_ids),  "왼팔 조인트 이름 확인 필요"
assert all(jid > 0 for jid in right_joint_ids), "오른팔 조인트 이름 확인 필요"

qidx_l = [model.idx_qs[jid] for jid in left_joint_ids]
qidx_r = [model.idx_qs[jid] for jid in right_joint_ids]

print("\n왼팔 7개 → 오른팔 7개 순서로 각도를 입력하세요. (단위: rad)")
print("예) 왼팔:  0 0 0 0 0 0 0")
print("    오른팔: 0 0 0 0 0 0 0")

while True:
    try:
        raw_l = input("\n[왼팔] 각도 7개 입력: ").strip()
        vals_l = list(map(float, raw_l.split()))
        if len(vals_l) != 7:
            print("왼팔은 정확히 7개 입력하세요.")
            continue

        raw_r = input("[오른팔] 각도 7개 입력: ").strip()
        vals_r = list(map(float, raw_r.split()))
        if len(vals_r) != 7:
            print("오른팔은 정확히 7개 입력하세요.")
            continue

        # q 벡터 구성
        q = pin.neutral(model)
        for idx, v in zip(qidx_l, vals_l):
            q[idx] = v
        for idx, v in zip(qidx_r, vals_r):
            q[idx] = v

        # FK & 프레임 업데이트
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        ee_pos_l = data.oMf[fid_l].translation
        ee_pos_r = data.oMf[fid_r].translation

        print(f"[L] EE (x,y,z): {ee_pos_l}")
        print(f"[R] EE (x,y,z): {ee_pos_r}")

    except KeyboardInterrupt:
        print("\n종료합니다.")
        break
    except Exception as e:
        print(f"[오류] {e}")
