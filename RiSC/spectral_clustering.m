function [clust_label, n_clust]=spectral_clustering(L, D, sc_type)
%%     Function to apply spectral clustering on the similarity graph
%
%       Parameters
%       ----------
%       L : n_node x n_node array
%           Graph Laplacian
%       D : n_node x n_node array
%           Degree matrix
%       sc_type : {"unnorm", "norm"}
%           Defines unnormalized or normalized spectral clustering (sc)
%
%       Returns
%       -------
%       clust_label : n_node x 1 array
%           Cluster label for each graph node
%       n_clust : doubles
%           Number of clusters
%
% Author: Maria Sayu Yamamoto (2023)
% <maria-sayu.yamamoto@universite-paris-saclay.fr>


% Apply eigenvalue decomposition on unnormalized graph Laplacian
if sc_type=="unnorm"
    [L_eigvec,L_eigvalue]=eig(L); % Standard eigenvalue decomposition
elseif sc_type=="norm"
    [L_eigvec,L_eigvalue]=eig(L, D); % Generalized eigenvalue decomposition
end

% Make sure the eigenvalue is aligned in ascending order
[L_eigvalue,ind]=sort(diag(L_eigvalue),'ascend');

% Eigengap heuristic
eigengaps = zeros(length(L_eigvalue)-1,1);
for i=1:1:length(L_eigvalue)
    if i > 5 % Maximum possible number of clusters is set to 5
        eigengaps(i)=-1;
    else
        eigengaps(i)=L_eigvalue(i+1)-L_eigvalue(i);
    end
end

% Determine the number of clusters
[~, k] = max(eigengaps);
n_clust = k;
disp(['n_clust = ' num2str(k)]);

% Define matrix U which contains the first k eigenvectors
u=L_eigvec(:,1:ind(k));

% Apply k-means to matrix U
[clust_label, ~] = kmeans(u,n_clust,'Replicates',200);

end
