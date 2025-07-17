# 基于低光照增强的车辆检测

## 环境配置
1. 创建conda环境

```
conda create --name LLC python=3.10
conda activate LLC
```

2. 安装pytorch

```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

3. 安装相关依赖

```
pip install -r requirements.txt
```

## LLC包使用说明
1. 如main函数所示，直接使用ALL()即可
2. 输入：numpy形式，存储在demo/pic中
3. 输出：numpy形式，存储在demo/exp中
4. 路径/模型等需要进入start中对对应参数的default进行修改
