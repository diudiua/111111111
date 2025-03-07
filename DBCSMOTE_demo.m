clc;
clear;

X1=importdata('train.txt');
validateset=importdata('validate.txt');
test=importdata('test.txt');

F=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15];
M = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]; %10,11,12 肌酐，ALT肝脏功能， LA左房内径  
attr_type=[1,0,0,0,1,1,1,1,1,0,0,0,1,1,0,0]; %对应列变量的属性
   
Out = [16];
age = X1(:,2)/90;
height = X1(:,3)/180;
weight = X1(:,4)/80;
vale = X1(:,6)/20;
scr = X1(:,10)/150;
alt = X1(:,11)/200;
la = X1(:,12)/90;
inr = X1(:,15)/3;

X=[X1(:,1),age,height,weight,X1(:,5),vale,X1(:,7:9),scr,alt,la,1.5*X1(:,13),1.2*X1(:,14),inr];

num = 100;  %循环次数
ES_representation = cell(1,num); % 建立结构体数组 choromsome;

train = X1;
[n , ~] = size(train);
m=round(n/10); %从多数样本中抽取m个样本与少样本做合成
 
tmp=10000;

for itr = 1:num
   ES_representation{1,itr}.epsilon= 0.96+rand(1,1)/10; %1.0-1.1
   ES_representation{1,itr}.MinPts=  round(rand(1,1)*3)+2 ;
  
   [mino_clusters,c] = DBSCAN_fun(X,X1, ES_representation{1,itr}.epsilon, ES_representation{1,itr}.MinPts);   
  
   ES_representation{1,itr}.ratio = 2*rand(1,c)+0.01; %k个采样率
   majorityneigbour_index = randperm(n, m);
   ES_representation{1,itr}.majority_part = train(majorityneigbour_index,:);
   ES_representation{1,itr}.mino_clusters = mino_clusters;
   ES_representation{1,itr}.c = c;

   [ES_representation{1,itr}]= ExtendTrainset(ES_representation{1,itr}, attr_type);
   
   [ES_representation{1,itr}, trainset_best, best_ind_index, sum_fitness] = evaluatePop(ES_representation{1,itr},num,train, validateset, F, Out);
   %predicted_largedose= solution_model.predicted_largedose;
   %observed_largedose = solution_model.observed_largedose;
   
   if tmp> ES_representation{1,itr}.score
       best_ind = itr;
       tmp = ES_representation{1,itr}.score;
   %    clear trainset_best;
   %    trainset_best = extrainset;
   end
  
end

%extended training set with DBCSMOTE
ind = ES_representation{1,best_ind};
%best_solution_model = ind.solution_model;

trainset = [train;ind.newSamples];
newsamples = [train;ind.newSamples];


%%%%%%%%%%%%%%%%%% train %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%  RF  %%%%%%%%%%%%%%%%%%%%%%%%

Trainset_input = trainset(:,F); %这里的trainset是扩展后的训练集
Trainset_output = trainset(:,Out);

Testset_input = test(:,F);
Testset_output = test(:,Out);

Validationset_input = validateset(:,F);
Validationset_output = validateset(:,Out);
treenum = 150;
%F=[1,2,3,4,5,6,7,8,9,13,14,15]; %特征的选择
feature_vector = F;  
Max_featurenum = 15;  %特征数量
%Max_featurenum = 12;
GA_choromsome.feature = feature_vector;

[DBCSMOTERF_evaluation, DBCSMOTERF_model,trees, validate_sum]= RandomForest(feature_vector, Trainset_input,Trainset_output,Testset_input,Testset_output,Validationset_input, Validationset_output, treenum, Max_featurenum);
%[solution_evaluation, solution_model]=brtdemo(F,Out,Trainset_input,Trainset_output,Testset_input,Testset_output)%性别	年龄	身高	体重	饮酒	胺碘酮	肌酐	ALT	目标INR CYP2CP*3 VKORC1 剂量(12)
view(trees{1}.tmp_tree);
