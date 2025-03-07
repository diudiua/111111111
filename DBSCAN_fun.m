function [min_cluster,k] = DBSCAN_fun(X,X1,epsilon,MinPts)
% 1 性别	2 年龄	3 身高	4 体重	5 瓣膜置换术后   6 充血性心衰	7 糖尿病	
% 8 饮酒	9 胺碘酮  10 肌酐  11 ALT  12 LA	 13 CYP2CP*3  14 VKORC1	15 目标INR	
% 16列标签： 剂量（天）
    
    [DX3,isnoise,D] = DBSCAN(X,epsilon,MinPts);
    
    %isnoise 是噪声点，DX3是所有点的所属聚类编号
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
