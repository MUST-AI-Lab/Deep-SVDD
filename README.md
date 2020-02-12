###实验思路
1. batchsize 一定，改变Focal 超参 ，每个类的结果变化 (只做了 0类  和7类的实验）
2. Focal 超参一定，改变batchsize ，每个类的结果变化
3. Batchsize，Focal 超参一定，要pretrain和不要，每个类的结果变化的区别 


*1. Focal 超参一定，改变batchsize ，每个类的结果变化

    只完成了 0 类  和  7 类
    

![Alt text](https://github.com/MUST-AI-Lab/Deep-SVDD/blob/master/imgs/class0.jpg)

![Alt text](https://github.com/MUST-AI-Lab/Deep-SVDD/blob/master/imgs/class7.jpg)








</br></br></br></br></br></br>
*3. Batchsize，Focal 超参一定，要pretrain和不要，每个类的结果变化的区别

    0, 1, 9 类，没有pre train 效果比用autoencoder 预训练好
    其余类相反。
    

![Alt text](https://github.com/MUST-AI-Lab/Deep-SVDD/blob/master/imgs/no%20pretrain.jpg)

![Alt text](https://github.com/MUST-AI-Lab/Deep-SVDD/blob/master/imgs/no.png)
