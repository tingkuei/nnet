clear;
clc;
Train=load('hw4_nnet_train.txt');
Test=load('hw4_nnet_test.txt');
M=[2 8 3 1];
repeat=500;
total_err=0;
for r=1:repeat
    r=r
    W=nnet_model(M,50000,Train,0.01,0.1);
    [err,accurancy] = nnet_predict(W,Test);
    accurancy=accurancy
    total_err=total_err+accurancy;
end
total_err=total_err/repeat;





