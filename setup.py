from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'new_bimanual_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # 기본 ROS 2 메타정보
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # MJCF 모델 포함시키기
        (os.path.join('share', package_name, 'mujoco_models'), 
            glob('mujoco_models/*.[ux]rdf') + glob('mujoco_models/*.xml')),
            
        # STL 파일들 포함 (assets 전체 복사)
        *[
            (
                os.path.join('share', package_name, root),
                glob(os.path.join(root, '*.stl'))
            )
            for root, _, _ in os.walk('mujoco_models/assets')
        ],

        # launch 파일도 있으면 포함 (선택)
        (os.path.join('share', package_name, 'launch'), 
            glob('launch/*.py')),
    ],
    install_requires=['setuptools', 'mujoco'],
    zip_safe=True,
    maintainer='gaga',
    maintainer_email='fhmpsy@gmail.com',
    description='MuJoCo + ROS2 interface',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'collision = new_bimanual_pkg.collision_detection:main',
            'linear = new_bimanual_pkg.linear_interpolation:main',
            'birrt = new_bimanual_pkg.birrt:main',
            'fk_node = new_bimanual_pkg.forward_node:main',
            'start_node = new_bimanual_pkg.start_node:main',
            'path_node = new_bimanual_pkg.path_node:main',
            'basic_node = new_bimanual_pkg.basic_spawn_node:main',
            'trajectory = new_bimanual_pkg.trajectory:main',
        ],
    },
)
