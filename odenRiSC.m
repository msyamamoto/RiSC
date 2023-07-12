function [clust_result, outlier_ind] = odenRiSC(g_type, sc_type, SCMs)
%%     Function to apply odenRiSC to a set of SCMs
%
%       Parameters
%       ----------
%       g_type : {"full", "knn"}
%           Defines the graph type
%       sc_type : {"unnorm", "norm"}
%           Defines unnormalized or normalized spectral clustering (sc)
%       SCM : 4-D array
%           Set of SCMs
%
%       Returns
%       -------
%       clust_result : 1 x n_cluster cell
%           Clustering results for SCM index
%       outlier_ind : doubles
%           Indices of detected outliers
%
% Author: Maria Sayu Yamamoto (2023)
% <maria-sayu.yamamoto@universite-paris-saclay.fr>


n_node=size(SCMs, 3);
SCM_ind=1:n_node;

%% ==== Apply RiSC====
[L, D]=creat_similarity_graph(g_type, SCMs);
[clust_label, n_clust]=spectral_clustering(L, D, sc_type);

%% ==== Determine outliers from RiSC outcome ====
M=mode(clust_label);
outlier_ind=find(clust_label~=M);

clust_result=cell(1, n_clust);
for clus=1:n_clust
    clust_result{clus}=SCM_ind(clust_label==clus);
end

end

