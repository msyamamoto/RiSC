function y_pred=mcRiSC_classifier_predict(clust_center, clust_class, SCM_test)
%%     Function to give predictions of mcRiSC classifier
%
%       Parameters
%       ----------
%       clust_center : 1xn_class cell
%           Centroid of each cluster
%       clust_class : 1d array
%           Class label of each cluster centroid
%       SCM_test :  1x1 struct
%           SCMs of testing set
%
%       Returns
%       -------
%       y_pred : 1d array
%           Predicted class label for each SCM
%
% Author: Maria Sayu Yamamoto (2023)
% <maria-sayu.yamamoto@universite-paris-saclay.fr>


% Compute the Riemannian distance between the given SCM and all cluster centroids.
dis=zeros(size(SCM_test, 3), size(clust_class, 2));
for clust=1:size(clust_class, 2)
    for i=1:size(SCM_test, 3)
        dis(i, clust)=distance_riemann(clust_center{clust}, SCM_test(:,:,i));
    end
end
% Select the cluster centroid with the minimum distance to each test data.
[~, ind]=min(dis,[],2);

% Store predicted labels
y_pred=zeros(1, size(SCM_test, 3));
for i=1:size(SCM_test, 3)
    if ind(i, 1) < find(clust_class == 2)
        y_pred(1, i)=1;
    else
        y_pred(1, i)=2;
    end
end
end