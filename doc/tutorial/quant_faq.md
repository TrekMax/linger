## 通用注意事项

1. 两阶段训练：
    - 量化训练通常分为浮点阶段和定点阶段
    - 浮点阶段：在原始浮点网络上进行约束训练，目的是让网络参数和激活分布更适合后续量化
    - 定点阶段：在浮点约束训练好的网络上进行量化训练，完成量化参数的学习和微调

2. 学习率设置：
    - 浮点阶段学习率通常设高一点，定点阶段学习率设低一点

## 常见问题定位与解决

1. 环境安装问题
    问题表现：error while loading shared libraries: libmpfr.so.6: cannot open shared object file: No such file or directory
    解决方案：将系统中 libmpfr.so.6 拷贝到环境中，例如：cp libmpfr.so /home4/listenai/miniconda3/envs/linger3.0/lib

## 网络搭建推荐

1. 搭建网络时，若 nn.Linear 层后有 bn 层直连，推荐将此 linear 层改为等效的 1*1 的卷积，这样在最后推理实现时可以完成 conv-bn 的融合，加快推理效率
2. 若网络中 conv-bn 的直连结构较多，使用 linger 进行量化时，推荐使用 intx 的 trace_layer 设置，完成 convbn 的自动融合，导图时即只会有一个 conv 存在
3. 网络 forward 代码中推荐使用  tensor.transpose/ tensor.squeeze / tensor.unsqueeze 等写法，其等效的 torch.transpose(temnsor, ...) 等写法可能暂时不被 linger 识别，导图时会出现量化 op 浮点的断连情况
4. 网络定义中推荐使用 nn.Conv2d / nn.Linear/ nn.BatchNorm 等写法，其等效的在 forward 中直接调用的 F.Conv2d 等写法可能暂时不被 linger 识别，导图时会出现 op 未量化的断连情况