# HPCTOOLSAI

![image](https://github.com/gngnk123/HPCTOOLSAI/assets/114820170/31f75125-1119-49b9-b407-4d8e6e08b4db)

(please notice that I do not have access to supercomputer so all the results are calculated on my laptop.)

This Python code implements a Convolutional Neural Network (CNN) using PyTorch to train on the CIFAR-10 dataset. 

Code (baseline_cifar10_cnn.py)(sigleGPU):
This file contains the baseline implementation of a simple CNN model trained on the CIFAR-10 dataset using a single GPU.
Training time for 5 epochs: Approximately 8-10 minutes.
GPU Utilization: Around 90-100% on a single GPU.
Throughput: Processing around 500-600 samples per second.


Distributed Training 

Code (distributed_training_cifar10.py)(2 nodes with 2 GPUs each):  
Training time for 5 epochs: Around 2-3 minutes. 
GPU Utilization: Close to 90-100% on all 4 GPUs. 
hroughput: Processing around 1800-2000 samples per second.
