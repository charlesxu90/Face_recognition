%% Run SVD on faces
supportedidxs = find(support_small);

B = zeros(length(supportedidxs), length(directory_list));

for i=1:length(directory_list)
  fprintf('reading %s \n', ['../Data/supported/' directory_list(i).name]);
  I = pgmRead(['../Data/supported/' directory_list(i).name]);

  B(:,i) = I(supportedidxs);
end

fprintf('Vectors stuffed in B.\n');

B_mean = mean(B')';
B_ = B-repmat(B_mean, 1,size(B,2));
[u_faces,s_faces,v_faces] = svd(B_,0);
