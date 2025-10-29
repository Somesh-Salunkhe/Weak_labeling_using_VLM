# Installation Guide
Clone repository:

```
git clone git@github.com:mariusbock/weak_har.git
cd weak_har
```

Create [Anaconda](https://www.anaconda.com/products/distribution) environment:

```
conda create -n weak_har python==3.11.7
conda activate weak_har
```

Install PyTorch distribution:

```
conda install pytorch==2.2.1 torchvision==0.17.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

Install other requirements:
```
pip install -r requirements.txt
```
