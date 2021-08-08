pink = [1 0.4 0.6];
teal = [0.1 0.9 0.8];

load('uspsdata.mat')
training_labels = uspstrain(:,1);
training_features = uspstrain(:,2:end);

testing_labels = uspstest(:,1);
testing_features = uspstest(:,2:end);

%the training set contains x and the test set contains y
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tr_zeroes_ref = find(training_labels==0);
tr_ones_ref = find(training_labels==1);
tr_twos_ref = find(training_labels==2);
tr_threes_ref = find(training_labels==3);
tr_fours_ref = find(training_labels==4);
tr_fives_ref = find(training_labels==5);
tr_sixes_ref = find(training_labels==6);
tr_sevens_ref = find(training_labels==7);
tr_eights_ref = find(training_labels==8);
tr_nines_ref = find(training_labels==9);

ts_zeroes_ref = find(testing_labels==0);
ts_ones_ref = find(testing_labels==1);
ts_twos_ref = find(testing_labels==2);
ts_threes_ref = find(testing_labels==3);
ts_fours_ref = find(testing_labels==4);
ts_fives_ref = find(testing_labels==5);
ts_sixes_ref = find(testing_labels==6);
ts_sevens_ref = find(testing_labels==7);
ts_eights_ref = find(testing_labels==8);
ts_nines_ref = find(testing_labels==9);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tr_zeroes_freq = length(tr_zeroes_ref);
tr_ones_freq = length(tr_ones_ref);
tr_twos_freq = length(tr_twos_ref);
tr_threes_freq = length(tr_threes_ref);
tr_fours_freq = length(tr_fours_ref);
tr_fives_freq = length(tr_fives_ref);
tr_sixes_freq = length(tr_sixes_ref);
tr_sevens_freq = length(tr_sevens_ref);
tr_eights_freq = length(tr_eights_ref);
tr_nines_freq = length(tr_nines_ref);

ts_zeroes_freq = length(ts_zeroes_ref);
ts_ones_freq = length(ts_ones_ref);
ts_twos_freq = length(ts_twos_ref);
ts_threes_freq = length(ts_threes_ref);
ts_fours_freq = length(ts_fours_ref);
ts_fives_freq = length(ts_fives_ref);
ts_sixes_freq = length(ts_sixes_ref);
ts_sevens_freq = length(ts_sevens_ref);
ts_eights_freq = length(ts_eights_ref);
ts_nines_freq = length(ts_nines_ref);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

training_freq = [tr_zeroes_freq, tr_ones_freq, tr_twos_freq, tr_threes_freq, tr_fours_freq...
    tr_fives_freq, tr_sixes_freq, tr_sevens_freq, tr_eights_freq, tr_nines_freq];

testing_freq = [ts_zeroes_freq, ts_ones_freq, ts_twos_freq, ts_threes_freq, ts_fours_freq...
    ts_fives_freq, ts_sixes_freq, ts_sevens_freq, ts_eights_freq, ts_nines_freq];

figure();
pie(training_freq,[],string(training_freq));
legend('0','1','2','3','4','5','6','7','8','9');
title('Number Frequency in Training Set')

figure();
pie(testing_freq,[],string(testing_freq));
legend('0','1','2','3','4','5','6','7','8','9');
title('Number Frequency in Testing Set')


%use above to find frequencies of each number for plotting in pie chart etc
%(as per Q1)

tr_ones = training_features(tr_ones_ref,:);
tr_twos = training_features(tr_ones_ref,:);
tr_threes = training_features(tr_ones_ref,:);
tr_fours = training_features(tr_ones_ref,:);
tr_fives = training_features(tr_ones_ref,:);
tr_sixes = training_features(tr_ones_ref,:);
tr_sevens = training_features(tr_ones_ref,:);
tr_eights = training_features(tr_ones_ref,:);
tr_nines = training_features(tr_ones_ref,:);
tr_tens = training_features(tr_ones_ref,:);

ts_ones = testing_features(ts_ones_ref,:);
ts_twos = testing_features(ts_ones_ref,:);
ts_threes = testing_features(ts_ones_ref,:);
ts_fours = testing_features(ts_ones_ref,:);
ts_fives = testing_features(ts_ones_ref,:);
ts_sixes = testing_features(ts_ones_ref,:);
ts_sevens = testing_features(ts_ones_ref,:);
ts_eights = testing_features(ts_ones_ref,:);
ts_nines = testing_features(ts_ones_ref,:);
ts_tens = testing_features(ts_ones_ref,:);

%visualise 4 random numbers from the dataset
figure();
for i =1:4
    subplot(2,2,i);
    imagesc(reshape(training_features(randi(max(size(training_features))),:),16,16)');    
end

%Q2
[coeff,score,latent] = pca(training_features);

figure();
plot(score(tr_zeroes_ref,1),score(tr_zeroes_ref,2),'*','MarkerSize',9);
hold on
plot(score(tr_ones_ref,1),score(tr_ones_ref,2),'*','MarkerSize',9);
plot(score(tr_twos_ref,1),score(tr_twos_ref,2),'*','MarkerSize',9);
plot(score(tr_threes_ref,1),score(tr_threes_ref,2),'*','MarkerSize',9);
plot(score(tr_fours_ref,1),score(tr_fours_ref,2),'*','MarkerSize',9);
plot(score(tr_fives_ref,1),score(tr_fives_ref,2),'*','MarkerSize',9);
plot(score(tr_sixes_ref,1),score(tr_sixes_ref,2),'*','MarkerSize',9);
plot(score(tr_sevens_ref,1),score(tr_sevens_ref,2),'k*','MarkerSize',9);
plot(score(tr_eights_ref,1),score(tr_eights_ref,2),'*','Color', pink,'MarkerSize',9);
plot(score(tr_nines_ref,1),score(tr_nines_ref,2),'*','Color',teal,'MarkerSize',9);
legend('0','1','2','3','4','5','6','7','8','9');

%Using Kmeans and comparing measures of similarity
%use chi2 to compare better
[cluster_id, centroids] = kmeans(training_features,10, 'Distance', 'sqeuclidean');
sqeuclidean_crosstab = crosstab(cluster_id, training_labels);

[cluster_id, centroids] = kmeans(training_features,10, 'Distance', 'cityblock');
cityblock_crosstab = crosstab(cluster_id, training_labels);

[cluster_id, centroids] = kmeans(training_features,10, 'Distance', 'cosine');
cosine_crosstab = crosstab(cluster_id, training_labels);

[cluster_id, centroids] = kmeans(training_features,10, 'Distance', 'correlation');
correlation_crosstab = crosstab(cluster_id, training_labels);

%visualising centroids
figure();
[cluster_id, centroids] = kmeans(training_features,10);
for i=1:10   
   subplot(2,5,i);
   imagesc(reshape(centroids(i,:),16,16)'); 
end


%answers may be different in subsequent runs - heuristic algorithm, likely
%to only find local minima to optimisation problem, not global minima.
%maybe compare accuracy in successive runs too
[cluster_id_1, centroids] = kmeans(training_features,10);
[cluster_id_2, centroids] = kmeans(training_features,10);

[crosstab_1, chi2_1] = crosstab(cluster_id_1, training_labels);
[crosstab_2, chi2_2] = crosstab(cluster_id_2, training_labels);
[crosstab_3, chi2_3] = crosstab(cluster_id_1, cluster_id_2);


%question 4, get all images of one class and a random selection and do MSD

random_sample_refs = randperm(max(size(training_features)),max(size(tr_ones)));
random_sample = training_features(random_sample_refs,:);

%change to squared euclidean perhaps - ask in lecture
pdist_ones = pdist(tr_ones);
pdist_rand_sample = pdist(random_sample);

MSD_ones = mean(pdist_ones);

figure();
title('Pair-Wise Distance Comparison for Five Random Subsets')
hold on
for i = 1:50
    random_sample_refs = randperm(max(size(training_features)),max(size(tr_ones)));
    random_sample = training_features(random_sample_refs,:);
    pdist_rand_sample = pdist(random_sample);
    MSD_rand_samples(i) = mean(pdist_rand_sample);
    histogram(pdist_rand_sample);   
end
legend('Random Sample 1','Random Sample 2','Random Sample 3','Random Sample 4','Random Sample 5')
hold off

figure();
histogram(pdist_ones,'FaceColor','r');
hold on
histogram(pdist_rand_sample,'FaceColor','b');
legend('pdist for Ones','pdist for Rand Sample');
xlabel('Pair-Wise Distance')
ylabel('Frequency')
title('Comparison Between Pair-Wise Distances for Random Sample and Ones Class')

figure();
histogram(MSD_rand_samples,'FaceColor','b');
title('Histogram for MSD of Random Subsets')
ylabel('Frequency')
xlabel('MSD')

%conclusion - MSD for one class is much lower on average, the points are
%much closer together therefore the images are much more similar


%Question 5 - decision tree fitting
%fitting and separating all 10 classes
Tree = fitctree(training_features,training_labels);

ts_Treepredictions = predict(Tree,testing_features);
tr_Treepredictions = predict(Tree,training_features);

ts_Treeconmat = confusionmat(ts_Treepredictions,testing_labels);
tr_Treeconmat = confusionmat(tr_Treepredictions,training_labels);

ts_Treeaccuracy = sum(diag(ts_Treeconmat))/sum(sum(ts_Treeconmat));
tr_Treeaccuracy = sum(diag(tr_Treeconmat))/sum(sum(tr_Treeconmat));

%separate class 5 and do a 2 class problem
two_class_training_labels = training_labels;
two_class_training_labels(find(two_class_training_labels ~= 5)) = 1;

two_class_testing_labels = testing_labels;
two_class_testing_labels(find(two_class_testing_labels ~= 5)) = 1;

%decision tree classifier
tree_two_class = fitctree(training_features, two_class_training_labels);
ts_Treepredictions_two_class = predict(tree_two_class, testing_features);
tr_Treepredictions_two_class = predict(tree_two_class, training_features);

two_class_ts_Treeconmat = confusionmat(ts_Treepredictions_two_class,two_class_testing_labels);
two_class_tr_Treeconmat = confusionmat(tr_Treepredictions_two_class,two_class_training_labels);

% repeat above using a SVM learner - check classification.m
SVMhyperplane = fitclinear(training_features,two_class_training_labels,'Learner','svm');
ts_SVMpredictions = predict(SVMhyperplane,testing_features);
tr_SVMpredictions = predict(SVMhyperplane,training_features);

two_class_ts_SVMconmat = confusionmat(ts_SVMpredictions,two_class_testing_labels);
two_class_tr_SVMconmat = confusionmat(tr_SVMpredictions,two_class_training_labels);

ts_SVMaccuracy = sum(diag(two_class_ts_SVMconmat))/sum(sum(two_class_ts_SVMconmat));
tr_SVMaccuracy = sum(diag(two_class_tr_SVMconmat))/sum(sum(two_class_tr_SVMconmat));

%Q6 - repeat above using knn
KNN = fitcknn(training_features,two_class_training_labels);

ts_KNNpredictions = predict(KNN,testing_features);
tr_KNNpredictions = predict(KNN,training_features);

ts_KNNconmat = confusionmat(ts_KNNpredictions,two_class_testing_labels);
tr_KNNconmat = confusionmat(tr_KNNpredictions,two_class_training_labels);

ts_KNNaccuracy = sum(diag(ts_KNNconmat))/sum(sum(ts_KNNconmat));
tr_KNNaccuracy = sum(diag(tr_KNNconmat))/sum(sum(tr_KNNconmat));



