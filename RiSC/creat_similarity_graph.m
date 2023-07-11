function [L, D]=creat_similarity_graph(g_type, SCM)
%%     Function to create knn or full-connected similarity graph 
%
%       Parameters
%       ----------
%       g_type : {"full", "knn"}
%           Defines the graph type
%       SCM : 4-D array
%           Set of SCMs
%           
%
%       Returns
%       -------
%       L : n_node x n_node array
%           Graph Laplacian
%       D : n_node x n_node array
%           Degree matrix
%
% Author: Maria Sayu Yamamoto (2023)
% <maria-sayu.yamamoto@universite-paris-saclay.fr>


node=SCM; % node : Graph node
n_node = size(node, 3); % Number of node

%% ====Set parameter theta(q) by MST====
[x,y] = find((triu(ones(n_node), 1))');
node_pair_list = [x, y];
dis = zeros(size(node_pair_list, 1), 1);

for ii = 1:size(node_pair_list,1)
    dis(ii, 1)=distance_riemann(node(:,:,x(ii)),node(:,:,y(ii)));
end

G = graph(x,y,dis);
[T,~] = minspantree(G,'Method','sparse', 'Type', 'tree');
theta = median(T.Edges.Weight);


%% ====Create a similarity graph====
if g_type=="knn"

    % Decide number of neighber k
    knn_k=round(log(n_node));

    % Initialize the adjacency matrix
    W=zeros(n_node);

    for i=1:size(node, 3)
        dis_set=[];
        A=node(:, :, i); % A : Center of knn
        for j=1:size(node, 3)
            % Compute Riemannian distance between the center of knn and all nodes
            temp=distance_riemann(A,node(:,:,j));
            dis_set=cat(1, dis_set, temp);
        end
        [~, ind]=sort(dis_set);
        knn_ind=ind(2:knn_k+1)';
        % If node i is knn of A , add 1 to the corresponding element 
        % in the adjacency matrix
        W(i, knn_ind)=1;
        W(knn_ind, i)=1;
    end

    % Add simirality weight on knn graph
    [row,col] = find((triu(W, 1))'); 
    for i=1:size(row, 1)
        % Compute pairwise Riemannian distance
        temp=distance_riemann(node(:,:,col(i)), node(:,:,row(i)));
        weight=exp(-(temp.^2)./(2*theta^2));
        W(col(i), row(i))=weight;
        W(row(i), col(i))=weight;
    end

    % Check if the given adjacency matrix is symmetric
        if issymmetric(W)==1
            disp('knn graph is created sucessifully');
        else
            warning('the given knn graph is not symmetric');
        end
   


elseif g_type=="full"
    [x,y] = find((triu(ones(n_node), 1))');
    node_pair_list = [x, y];
    dis = zeros(size(node_pair_list, 1), 1);

    % Compute Riemannian distances between all graph nodes
    for ii = 1:size(node_pair_list,1)
        dis(ii, 1)=distance_riemann(node(:,:,x(ii)),node(:,:,y(ii)));
    end

    % Create a weighted adjacency matrix
    E = exp(-(dis.^2)./(2*theta^2));
    W = squareform(E);
end


%% ====Create degree matrix and unnormalized graph Laplacian====
D = diag(sum(W,2));
L = D-W;

end

