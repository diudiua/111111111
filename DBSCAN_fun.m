function [min_cluster,k] = DBSCAN_fun(X,X1,epsilon,MinPts)
% 1 �Ա�	2 ����	3 ���	4 ����	5 ��Ĥ�û�����   6 ��Ѫ����˥	7 ����	
% 8 ����	9 ����ͪ  10 ����  11 ALT  12 LA	 13 CYP2CP*3  14 VKORC1	15 Ŀ��INR	
% 16�б�ǩ�� �������죩
    
    [DX3,isnoise,D] = DBSCAN(X,epsilon,MinPts);
    
    %isnoise �������㣬DX3�����е������������
    maxindex  = max(max(DX3));
    k=1;
    if maxindex>3
        min_cluster = cell(1,maxindex-2);
    elseif maxindex==3
        min_cluster = cell(1,2);
    else
        min_cluster = cell(1,1);
    end
    if maxindex>3
        for i = maxindex:-1:4
            min_cluster{1,k} = X1(DX3==i,:);
            k = k+1;
        end
        min_cluster{1,k} = X1(DX3==0,:);
    elseif maxindex==3
        min_cluster{1,k}=X1(DX3==maxindex,:);
        k=k+1;
        min_cluster{1,k} = X1(DX3==0,:);
    else
        min_cluster{1,1} = X1(DX3==0,:);
    end
    
end
