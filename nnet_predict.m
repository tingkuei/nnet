function [err,accurancy] = nnet_predict(W,Data)
    layer=size(W,2);
    out_size=size(Data,1);
    M=[];
    for i =1:layer
        M=[M size(W{1,i},1)-1];
    end
    M=[M 1];
    err=0;
    for i=1:out_size
        S_E=cell(1,layer+1);
        input=Data(i,:);
        yn=input(size(Data,2));
        input=input(1:size(Data,2)-1);
        S_E{1,1}=[1; input'];
        for k=1:layer
            back_num=M(k+1);
            w=W{1,k};
            s=[];
            B=S_E{1,k};
            for j=1:back_num
                A=w(:,j);
                score=tanh(A'*B);
                s=[s score];
            end
            if k == layer
                if sign(s) ~= yn
                    err=err+1;
                end
            else
                s=[1 s];
                S_E{1,k+1}=s';
            end
        end
    end
    err=err;
    accurancy=err/out_size;
end

