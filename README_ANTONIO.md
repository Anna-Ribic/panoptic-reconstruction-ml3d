# Setup

- conda env create --file environment.yaml
    - **without torch, cudatoolkit and requirements.txt**
- Install cuda manually (11.1)
- `pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html`
- `sudo apt-get install libopenexr-dev`
- `pip install -r requirements.txt`

---

MinkowskiEngine

- Download from Nvidia’s repo (not forked repo)
- `conda install openblas-devel -c anaconda`
- `export CUDA_HOME=/usr/local/cuda-11.1`
- `make clean`
- `python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas`

---

```
`cd lib/csrc/`
`python setup.py install`
```

---

```python
git clone https://github.com/ptrblck/apex.git
cd apex
git checkout apex_no_distributed
pip install -v --no-cache-dir ./
```

---

```python
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt install -y g++-11

strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
```

---

```python
git clone https://github.com/pdollar/coco.git

pip install Cython

cd coco/PythonAPI
make
make install
python setup.py install
```

---

Replace “np.float)” in code with “float)” in code

---
Downlaod the necessary missing files from the repo (in the table) - you might have to rename downloaded model
```python
cd data
wget https://kaldir.vc.in.tum.de/panoptic_reconstruction/panoptic-front3d.pth
rename it panoptic_front3d.pth
```

---


---
#FIX EXTENSION ERROR when installing mask-rcnn

cuda_dir="maskrcnn_benchmark/csrc/cuda"

perl -i -pe 's/AT_CHECK/TORCH_CHECK/' $cuda_dir/deform_pool_cuda.cu $cuda_dir/deform_conv_cuda.cu


# You can then run the regular setup command
python3 setup.py build develop



python test_net_single_image.py -i data/front3d-sample/rgb_0007.png -o outputs/front3d-sample-0007
