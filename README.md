# SeqCSGM
Compressed sensing of sequential signal by leveraging generative models
# setup  
## Requirements:
1. Python 3
2. torch 2.6.0
3. numpy 2.0.2
4. osqp  0.6.7.post3
5. matplotlib
6. scipy
## Download/extract the datasets:
* UCF101: [https://www.crcv.ucf.edu/research/data-sets/ucf101/]()
* Moving mnist: [https://www.cs.toronto.edu/~nitish/unsupervised_video/]()
  
Place the datasets in the “data” folder: data/UCF101, data/Movingmnist
## Download the models
The pretrained vae model: [https://huggingface.co/maxin-cn/Latte]() or [https://pan.baidu.com/s/1fY74z2GSdu4qo1vFjp6QUQ?pwd=i4cf]()
The end2end model also can trained in e2e_train.ipynb

Place the models in the "models" folder
# Run the SeqCSGM and other recovery methods
python main.py --dataset ucf --batch_size 100 --llambda 10
