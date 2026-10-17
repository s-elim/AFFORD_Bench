import shutil
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent


class BuildPyWithConfigs(build_py):
    """Ship configs/prompts_augmented.yaml inside the package (task tiers and instruction variants need it)."""

    def run(self):
        super().run()
        target = Path(self.build_lib) / "afford_pro" / "configs"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "configs" / "prompts_augmented.yaml", target / "prompts_augmented.yaml")


setup(
    name="afford_pro",
    version="0.1.0",
    license="MIT",
    description="AFFORD-Bench: benchmark, oracle dataset generator, and OOD evaluation suite for robot manipulation policies",
    author="Selim Sarowar & Antigravity",
    packages=find_packages(include=["afford_pro", "afford_pro.*"]),
    package_data={"afford_pro": ["assets/*.xml", "assets/robots/assets/*", "assets/robots/LICENSE",
                                 "assets/robots/NOTICE.md", "assets/textures/*"]},
    cmdclass={"build_py": BuildPyWithConfigs},
    python_requires=">=3.10",
    install_requires=[
        # Every reported number was measured with this simulator build; contact results change across MuJoCo releases.
        "mujoco==3.5.0",
        "numpy>=1.22.0",
        "h5py>=3.8.0",
        "imageio>=2.28.0",
        "imageio-ffmpeg",
        "opencv-python>=4.7.0",
        "pyyaml>=6.0",
        "gymnasium>=0.28.0",
    ],
    extras_require={"learning": ["torch>=2.1"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
    ],
)
