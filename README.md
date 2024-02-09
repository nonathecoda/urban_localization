# urban_localization

conda activate envp39
source venv/bin/activate


## How to install Open3D on ARM Mac

https://github.com/isl-org/Open3D/releases/tag/v0.14.1

1. wget https://github.com/isl-org/Open3D/releases/download/v0.14.1/open3d-0.14.1-cp39-cp39-macosx_11_0_arm64.whl
2. conda create -n envp39 python=3.9
3. conda activate envp39
4. python -m venv /Users/antonia/dev/masterthesis/render_hilla/v2.0/venv
5. source venv/bin/activate
6. pip install open3d-0.14.1-cp39-cp39-macosx_11_0_arm64.whl

double check with python --version if its 3.9

Then, pip install -r requirements.txt