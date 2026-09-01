from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'd435i_multicam_launch'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # Required for ament index
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),

        # Package manifest
        ('share/' + package_name, ['package.xml']),

        # 🔴 THIS IS THE IMPORTANT PART (install launch files)
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nithul',
    maintainer_email='nithul@todo.todo',
    description='Launch package for 4x Intel RealSense D435i cameras',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [

            # The one command a session runs. Owns the bag.
            'record_session = d435i_multicam_launch.record_session:main',

            # Post-hoc check: what resolution did a bag actually record?
            'inspect_bag = d435i_multicam_launch.inspect_bag:main',

            # UNUSED: retired marker mechanism, kept for reference.
            'episode_marker = d435i_multicam_launch.episode_marker:main',
        ],
    },
)
