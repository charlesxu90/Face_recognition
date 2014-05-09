directory_list = dir('../faces/*.pgm');

support=pgmRead('../support.pgm')/255;

samplingR = 1:2:size(support,1);
samplingC = 1:2:size(support,2);
support = support(samplingR, samplingC);
pgmWrite(support, '../eigfaces/support.pgm', [0 1]);
supportedidxs = find(support);

B = zeros(length(supportedidxs), length(directory_list));

for i=1:length(directory_list)
  fprintf('\nappending supported/%s', directory_list(i).name);
  I=pgmRead(['../supported/' directory_list(i).name]);

  I = I(samplingR, samplingC);
  B(:,i) = I(supportedidxs);
end

B_mean = mean(B,2);
B_ = B - B_mean*ones(1,size(B,2));
[u_faces,s_faces,v_faces] = svd(B_,0);
