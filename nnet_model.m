function W = nnet(M,iteration,Data,n,ra)
    % M determine network structure
    % iteration determine times for stochatis gradient decent
    % data for train input
    % n determine step size for SGD
    % ra for weight initial range
    
    layer=size(M,2)-1;
    train_size=size(Data,1);
    % initial weight
    for k=1:layer
        pre_num=M(k);
        back_num=M(k+1);
        clear w;
        for j=1:back_num
            for i=1:pre_num+1
                w(i,j)=rand*2*ra-ra;
            end
        end
        W{1,k}=w;
        clear w;
    end
    for T=1:iteration
        input=Data(randi(train_size,1,1),:);
        yn=input(size(Data,2));
        input=input(1:size(Data,2)-1);
        S{1,1}=[1; input'];
        % forward
        for k=1:layer
            pre_num=M(k);
            back_num=M(k+1);
            w=W{1,k};
            s=[];
            B=S{1,k};
            for j=1:back_num
                A=w(:,j);
                score=tanh(A'*B);
                s=[s score];
            end
            if k ~= layer
            s=[1 s];
            end
            S{1,k+1}=s';
            s=[];
        end
        %backward
        for k=layer:-1:1
            pre_num=M(k);
            back_num=M(k+1);
            s_b=S{1,k+1};
            s_p=S{1,k};
            w=W{1,k};
            if k == layer
                d=-2*(yn-s_b)*(1-s_b^2);
                for j=1:back_num
                    for i=1:pre_num+1
                        w(i,j)=w(i,j)-n*s_p(i)*d(j);
                    end
                end
                De{1,k}=d;
                W{1,k}=w;
            else
                d_b=De{1,k+1};
                w_b=W{1,k+1};
                n_last=M(k+2);
                d=[];
                for i=2:back_num+1
                    temp_sum=0;
                    for j=1:n_last
                        temp_sum=temp_sum+d_b(j)*w_b(i,j);
                    end
                    d(i-1)=temp_sum*(1-s_b(i)^2);
                end
                De{1,k}=d';
                for j=1:back_num
                    for i=1:pre_num+1
                        w(i,j)=w(i,j)-n*s_p(i)*d(j);
                    end
                end
                W{1,k}=w;
            end
        end
    end
end