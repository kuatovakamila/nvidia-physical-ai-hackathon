from setuptools import find_packages, setup

package_name = "within_reach"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Kamila Kuatova",
    maintainer_email="kamila.kuatova2025@gmail.com",
    description="Goal-stack kitchen assistant as a ROS 2 node.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "kitchen_node = within_reach.kitchen_node:main",
        ],
    },
)
