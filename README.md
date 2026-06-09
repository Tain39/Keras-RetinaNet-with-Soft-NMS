# Soft-NMS-Kares-RetinaNet
Reproduction of Keras RetinaNet and replacing native NMS post processing with Soft-NMS

 尝试复现 Keras RetinaNet.首先，我使用了一个小型且自定义的数据集，其中有 4 张图片用于对权重模型进行后训练，以确定环境是否正确，训练能否进行。当一切正常后，我转向 VOC2007 和 COCO2014 数据集正式训练我的模型。受设备限制，我将batch size设置为 1;对VOC和COCO分别进行了每轮 1000 步和 10000 步训练.
 
 另外， Keras RetinaNet 原生的项目文件目录中好像没有包含推理脚本.因此，我编写了一个使用原生 NMS 方法进行后处理的 "infer.py"，以及一个一个使用 Soft-NMS 方法的 "inference.py"。通过引入 Soft-NMS，我发现预测框的重叠得到了减少.
 
 根据 Keras RetinaNet 的论文，进行了一系列消融实验。与论文不同的是，我发现最佳超参数组合是 alpha=0.25，gamma=1，并配合锚框 #sc2-ar3, 结果的差异可能源于设备差异和训练设置.
 
下载Kares-RetinaNet项目文件后，将推理脚本拖入文件目录即可.

 A trial of reproducing Keras RetinaNet. Firstly I uesd a small and customized dataset which has 4 pictures to post-train the weight model, in order to figure out whether the environment is correct or not. When everything was OK, I turned to VOC2007 and COCO2014 datasets to officially train my model. Limited by the device, I set the batch size to 1 with 1000 and 10000 steps each epoch respectively to VOC and COCO.
 
 It seems that the project files of Keras RetinaNet didn't contain the inference script. Thus I wrote an "infer.py" which post-processes with native NMS method, and  an "inference.py" using Soft-NMS method. And I noticed that the overlap of prediction boxes is reduced by introducing Soft-NMS.
 
 A series of ablation experiment are done indicated by the paper of Keras RetinaNet. Different with the paper, I found that the best combination of super parameters is alpha=0.25, gamma=1 accompanied with #sc2-ar3. The differences could come from device and training settings.
