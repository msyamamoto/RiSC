function [clust_center, clust_class]=mcRiSC_classifier_fit(clustered_SCM)
%%     Function to fit mcRiSC classifier to clustered training data
%
%       Parameters
%       ----------
%       clustered_SCM : 1x1 struct
%           clustered SCMs of each class
%
%       Returns
%       -------
%       clust_center : 1xn_class cell
%           Centroid of each cluster
%       clust_class : 1d array
%           Class label of each cluster centroid
%
% Author: Maria Sayu Yamamoto (2023)
% <maria-sayu.yamamoto@universite-paris-saclay.fr>

% Remove outlier clusters
for c=1:size(struct2cell(clustered_SCM), 1)
    for i=1:numel(clustered_SCM.(['class' num2str(c)]))
        if size(clustered_SCM.(['class' num2str(c)]){1, i},3)<2
            clustered_SCM.(['class' num2str(c)]){1,i}=[];
        end
    end
    % Update the clustered_SCM cell
    ind = cellfun(@isempty,clustered_SCM.(['class' num2str(c)]));
    clustered_SCM.(['class' num2str(c)])(ind)=[];
end

% Assign a class label to each cluster
all_class_clust=[];
clust_class=[];
for c=1:size(struct2cell(clustered_SCM), 1)
    all_class_clust=cat(2, all_class_clust, clustered_SCM.(['class' num2str(c)]));
    clust_class=cat(2, clust_class, c*ones(1, size(clustered_SCM.(['class' num2str(c)]), 2)));
end

% Estimate each cluster centroid
clust_center{1, size(all_class_clust, 2)}=[];
for clust=1:size(all_class_clust, 2)
    clust_center{1, clust}=riemann_mean(all_class_clust{clust});
end

end