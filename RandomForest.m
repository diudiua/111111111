function [solution_evaluation, solution_model,Trees,validate_sum]= RandomForest(feature_vector, TrainInput_Org,TrainOutput_Org,TstInput_Org,TstOutput, Validationset_input, Validationset_output, treenum, Max_featurenum)
 Trees = cell(1,treenum); % �����ṹ������ choromsome;
 [train_m,train_n]=size(TrainInput_Org);
 [tst_m,tst_n]=size(TstInput_Org);
 
 N=train_m; %��ȡtrain_m ȡ��
 

  
 cof_train=0;
 train_mae=0;
 %train_mape=0;
 train_mse=0;
 validate_sum=0;
 
 for i=1:treenum
  disp(i);
  
  %���������г���   
  %�������������������15
  s_point= Max_featurenum-3;
  feature_n = randi(Max_featurenum-s_point,1)+s_point; % ʵ�ʸ�ÿ�����ı�����Ϊ�� 7~Max_featurenum��
  feature_index_vector = randperm(feature_n);
  Trees{i}.feature = feature_vector(feature_index_vector);
  Trees{i}.tmp_tree = [];
  %Trees{i}.feature = feature_index_vector;
  %disp(Trees{i}.feature);
   
  %��ѵ�������г���
  train_instance_random = randperm(N); %randperm����������N�Ĳ��ظ�������
  %Sub = 5000
  
  %train_actual_n = train_end -train_start+1;
  XN = round(N*(0.4+0.6*rand(1,1))); %���ܵ���������������Ϊ(0.5+0.5*rand(1,1))��������ÿ������ѵ����
  train_row_index=train_instance_random(1:XN);
  %disp(train_row_index);
%  train_row_index=1:100;
  TrainInput = TrainInput_Org(train_row_index,:);
  TrainOutput = TrainOutput_Org(train_row_index,:);
  
  TstInput = TstInput_Org(:,:);
 
  %CART����Ҫ����Ҷ�ӽڵ�����  prediction
  [Trees{i}.solution_eva, Trees{i}.solution_mod, Trees{i}.tmp_tree] = CARTprediction(Trees{i}, TrainInput,TrainOutput,TstInput,TstOutput)
   % sum_train = sum_train +Trees{i}.solution_mod.train; 
   % sum_tst = sum_tst +Trees{i}.solution_mod.test
   %Trees{i}.tree = tmp_tree;
   
   %�������ʹ��Validationset_input(:,Trees{i}.feature)���������б�����������
   Cart_result_validation=predict(Trees{i}.tmp_tree,Validationset_input(:,Trees{i}.feature));%ʹ�ò�������������֤Validationset
   %ͨ��Validationset�ϵľ����������ɭ��������ϵ�Ȩ��
   v_e=(Validationset_output-Cart_result_validation);     %��֤�����ϵ����
   Trees{i}.validation_mse=(sum(v_e.^2))/50;
   validate_sum = validate_sum + (1/Trees{i}.validation_mse);
 
   %cof_train = cof_train + Trees{i}.solution_eva.trainR2;
   %train_mae = train_mae + Trees{i}.solution_eva.trainmae;
   %train_mse = train_mse + Trees{i}.solution_eva.trainmse;
   %train_mape = train_mape + Trees{i}.solution_eva.trainmape;
 end
 %sum_train = sum_train/treenum; %ģ����ѵ�������ϵ����������ƽ��ֵ��Ϊɭ�ֵĽ������Ϊÿ����ѵ��������������ͬ
 %sum_tst=sum_tst;
  %����Ȩ����������ɭ���ڲ��Լ����ϵ����
  tst_ensemble = zeros(tst_m,1);
  for i=1:treenum
     w=(1/Trees{i}.validation_mse)./validate_sum; 
     tst_ensemble =tst_ensemble+ w.*Trees{i}.solution_mod.test;  %����ģ���ڲ��Լ��ϵ�������ȡ��ƽ��
     train_mse = train_mse +  w.*Trees{i}.solution_eva.trainmse;
     train_mae = train_mae + w.*Trees{i}.solution_eva.trainmae;
     cof_train = cof_train + w.*Trees{i}.solution_eva.trainR2;
  end
   %TstOutput = TstOutput*100;
 %maxt = max(max(TrainOutput));
 max_t_test=max(max(TstOutput));

 %

 %corr
 cof_test = corr(TstOutput,tst_ensemble);
 %cof_train = cof_train/treenum;
  
 %max_t_test = 1
 %MAE
 %train_mae=train_mae/treenum;
 test_e=abs(TstOutput-tst_ensemble)/max_t_test;     %���Լ����ϵ����
 test_mae=sum(test_e)/tst_m; 
 
 %MSE%���Լ����ϵ����
 % train_mse= train_mse/treenum;
 test_e=(TstOutput-tst_ensemble)/max_t_test;     %���Լ����ϵ����
 test_mse=sum(test_e.^2)/tst_m;
 
  %MAPE
% ind = find(TrainOutput)
% Tout = TrainOutput(ind);
% Pout = sum_tst(ind)
% ind_test = find(TstOutput)
% Tout_test = TstOutput(ind_test);
% Pout_test = sum_tst(ind_test)
 
  %���Լ����ϵ����
% train_mape=train_mape/treenum;
% test_e=abs(TstOutput-sum_tst)./(TstOutput+0.01);     %���Լ����ϵ����
% test_mape=sum(test_e)/tst_m;
 
 %RMSE
%train_e=t_train-netout_trian;     %���Լ����ϵ����
 train_rmse=sqrt(train_mse);
%test_e=t_test-netout_test;     %���Լ����ϵ����
 test_rmse=sqrt(test_mse);
 
 %20%-p
 
   % num=0;
   %  for j=1:1:XN
   %     if train_ex(j)<0.2
   %        num = num+1;
   %     end
   %  end
   %  p_20_train = num/XN;
     
     test_e=abs(TstOutput-tst_ensemble)./(TstOutput+0.01); 
     num=0;
     for j=1:1:tst_m
        if test_e(j)<0.2
           num = num+1;
        end
     end
     p_20_test = num/tst_m;
     
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
 solution_evaluation.testp20 = p_20_test;
 
 solution_model.test = tst_ensemble; %����ģ��
end