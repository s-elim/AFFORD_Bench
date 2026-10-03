from setuptools import setup, find_packages

setup(
    name="afford_pro",
    version="0.1.0",
    description="AFFORD-PRO-150: Benchmark, Dataset Generator & OOD Evaluation Suite for VLA Models",
    author="Selim Sarowar & Antigravity",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "mujoco>=3.1.0",
        "numpy>=1.22.0",
        "h5py>=3.8.0",
        "imageio>=2.28.0",
        "imageio-ffmpeg",
        "opencv-python>=4.7.0",
        "pyyaml>=6.0",
        "gymnasium>=0.28.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
    ],
)
