%% Write out average face, eigenvalues, and eigenvectors
eigenvalues = diag(s_faces);
save ../Data/eigenvalues.dat eigenvalues -ascii;

supportedidxs = find(support_small);
I = zeros(size(support_small));
I(supportedidxs) = B_mean;
pgmWrite(I, '../Data/eigfaces-for-your-viewing-pleasure/average.pgm');
pgmWrite(support_small, '../Data/eigfaces/support.pgm');
fid = fopen('../Data/eigfaces/average.dat', 'wb');
fwrite(fid, I', 'float32');
fclose(fid);

for i=1:size(B,2)
  I(supportedidxs) = u_faces(:,i);

  fprintf('writing ../Data/eigfaces/%03d.dat\n', i);
  pgmWrite(I, sprintf('../Data/eigfaces-for-your-viewing-pleasure/%03d.pgm', i));

  fid = fopen(sprintf('../Data/eigfaces/%03d.dat', i), 'wb');
  fwrite(fid, I', 'float32');
  fclose(fid);
end
