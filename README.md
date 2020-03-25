# 설치방법
1. 라이브러리 설치  

```
conda create -n env python=3.6.9
conda activate env
conda install pytorch=1.2.0 torchvision cudatoolkit=10.0 -c pytorch
pip install tensorflow-gpu==1.6.0
pip install numpy==1.16.4 pygame networkx gym==0.15.4 PyYAML matplotlib

```
2.  환경설정

   .bashrc 파일에서

   ```
   export PATH=/usr/local/cuda-9.2/bin${PATH:+:${PATH}}
   export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
   ```

   환경변수를 등록해주고 재부팅한다.

3. 설치되었는지 확인  

   TensorFlow

```
import tensorflow as tf
print(tf.__version__)
tf.test.is_gpu_available(
    cuda_only=False,
    min_cuda_compute_capability=None)
```
​	PyTorch

```
import torch
torch.cuda.device_count()
torch.cuda.get_device_name(0)
torch.cuda.is_available()
```


# version

python == 3.6.9

pytorch == 1.2.0

tensorflow = 1.6.0

tensorflow-gpu == 1.6.0

cudatoolkit == 10.0

cudnn == 7

gcc == 4.8

bazel == 0.9.0

numpy==1.16.4

[호환성 참조](https://stackoverflow.com/questions/50622525/which-tensorflow-and-cuda-version-combinations-are-compatible)



# numpy version문제

```
import tensorflow
```

를 했을 때 

in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.) 같은 warning message가 출력된다면 numpy 1.17이 설치되는 경우 발생하는 문제로, 삭제 후 1.16.4 버전을 깔아주면 해결된다



# 참조문헌
[ddpg 코드](https://github.com/ghliu/pytorch-ddpg)
