supportedidxs = find(support);
[supr,supc] = find(support);
rmin=min(supr); rmax=max(supr); cmin=min(supc); cmax=max(supc);
support_small = support(rmin:rmax, cmin:cmax);

for i=1:length(directory_list)
  fprintf('supporting %s \n', ['../Data/unpadded/' directory_list(i).name]);
  I = pgmRead(['../Data/unpadded/' directory_list(i).name]);

  Io = I(rmin:rmax, cmin:cmax) .* support_small;

  pgmWrite(Io, ['../Data/supported/' directory_list(i).name], [0 255]);
end
