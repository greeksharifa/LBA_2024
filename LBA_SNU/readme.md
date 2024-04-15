# LBA_SNU

## Installation
```bash
# conda
conda create -n LBA_SNU python=3.8
conda activate LBA_SNU

# lavis
# 프로젝트 바깥 or 원하는 위치에 설치 가능
git clone https://github.com/salesforce/LAVIS.git
cd LAVIS
pip install -e .

# resolve version conflict
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install chardet
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless 
pip install opencv-python==4.7.0.72
pip install Triton==2.1.0
```