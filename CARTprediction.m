function [solution_evaluation, solution_model, tree]= CARTprediction(GA_choromsome, TrainInput_Org,TrainOutput,TstInput_Org,TstOutput)

%[train_m,train_n]=size(TrainInput_Org(:,GA_choromsome.feature));
%[test_m,test_n]=size(TstInput_Org(:,GA_choromsome.feature));

TrainInput= TrainInput_Org(:,GA_choromsome.feature);

TstInput = TstInput_Org(:,GA_choromsome.feature);

%构建CART算法分类树fitrtree 用于回归, fitctree用于分类
tree=fitrtree(TrainInput,TrainOutput,'MinLeafSize',12);   %MinLeafSize 每个节点最小的叶子数量，越小，树的深度就越大，越大，树的深度就越小
%  MinLeafSize 对叶子节点数量最小值做限制
%[~,~,~,bestlevel] = cvLoss(tree,...
%    'SubTrees','All','TreeSize','min') 剪支
%tree = prune(tree,'Level',bestlevel);
%view(tree,'Mode','graph');%生成树图

%rules_num=(tree.IsBranchNode==0);
%rules_num=sum(rules_num);%求取规则数量
Cart_result_train=predict(tree,TrainInput);%使用测试样本进行验证
Cart_result_test=predict(tree,TstInput);%使用测试样本进行验证
decision_values_train=Cart_result_train;
decision_values_tst = Cart_result_test;

%Cart_result=cell2mat(Cart_result);

      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%少数样本误差%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[largedose_index,~]=find(TrainOutput>4.2|TrainOutput<1.5);
predicted_largedose = Cart_result_train(largedose_index);
observed_largedose = TrainOutput(largedose_index);
largedose_max = max(max(observed_largedose));
[mino_n,~] = size(predicted_largedose);
largedose_e=(predicted_largedose-observed_largedose)/largedose_max;  
largedose_mse=(sum(largedose_e.^2)/mino_n); 
%[largedose_index,j~]=find(TrainOutput>5);
%Minority_set_input = TstInput(i,:);
%Minority_set_output = TstOutput(i,:);
%[mino_n,mino_x] = size(Minority_set_output);
%max_m = max(max(Minority_set_output));
%GA_choromsome.i为少数样本索引
%Minority_set_predictedoutput = Cart_result_test(GA_choromsome.mino_i,:);
%Minority_e=(Minority_set_output-Minority_set_predictedoutput)/max_m;     %测试集合上的误差
%minority_mse=(sum(Minority_e.^2)/mino_n); 
%solution_evaluation.minority_mse = minority_mse;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%总体样本误差%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



max_t_test=max(max(TstOutput));
maxt = max(max(TrainOutput));
 
[tr_n,tr_x] = size(TrainOutput);
[tst_n,tst_x] = size(TstOutput);
 
 %corr
 cof_test = corr(TstOutput,decision_values_tst);
 cof_train = corr(TrainOutput,decision_values_train);
  
 %MAE
 train_e=abs(TrainOutput-decision_values_train)/maxt;     %测试集合上的误差
 train_mae=sum(train_e)/tr_n;
 test_e=abs(TstOutput-decision_values_tst)/max_t_test;     %测试集合上的误差
 test_mae=sum(test_e)/tst_n; 

 %MSE
 train_e=(TrainOutput-decision_values_train)/maxt;     %测试集合上的误差
 %disp(maxt)
 train_mse=(sum(train_e.^2)/tr_n);
 test_e=(TstOutput-decision_values_tst)/max_t_test;     %测试集合上的误差
 test_mse=sum(test_e.^2)/tst_n;
 
  %MAPE
% ind = find(TrainOutput)
% Tout = TrainOutput(ind);
% Pout = decision_values_train(ind)
% ind_test = find(TstOutput)
% Tout_test = TstOutput(ind_test);
% Pout_test = decision_values_tst(ind_test)
 
 %train_e=abs(TrainOutput-decision_values_train)./(TrainOutput+0.01);     %测试集合上的误差
 %train_mape=(sum(train_e)/tr_n);
 %test_e=abs(TstOutput-decision_values_tst)./(TstOutput+0.01);     %测试集合上的误差
 %test_mape=sum(test_e)/tst_n;
 
 %RMSE
%train_e=t_train-netout_trian;     %测试集合上的误差
 train_rmse=sqrt(train_mse);
%test_e=t_test-netout_test;     %测试集合上的误差
 test_rmse=sqrt(test_mse);
 
 solution_evaluation.train_largedose_mse = largedose_mse;
 
 solution_evaluation.trainmse =train_mse;
 solution_evaluation.trainmae = train_mae;
% solution_evaluation.trainmape = train_mape;
 solution_evaluation.trainrmse = train_rmse;
 solution_evaluation.trainR2 = cof_train*cof_train;
 %solution_evaluation.trainp20 = p_20_train;
 solution_evaluation.testmse = test_mse;
 solution_evaluation.testmae = test_mae;
% solution_evaluation.testmape = test_mape;
 solution_evaluation.testrmse = test_rmse;
 solution_evaluation.testR2 = cof_test*cof_test;
 %solution_evaluation.testp20 = p_20_test;
 
 
 solution_model.train = decision_values_train;
 solution_model.predicted_largedose = predicted_largedose;
 solution_model.observed_largedose = observed_largedose;
 solution_model.test =decision_values_tst;
 %solution_model.Minority_set_predictedoutput =Minority_set_predictedoutput;
 
 % disp(max_t_test)
end
%Y_test=cell2mat(Y_test);

%Cart_result=(Cart_result==Y_test);
%Cart_length=size(Cart_result,1);%统计准确率
%Cart_rate=(sum(Cart_result))/Cart_length;

%disp(['规则数：' num2str(rules_num)]);
%disp(['测试样本识别准确率：' num2str(Cart_rate)]);