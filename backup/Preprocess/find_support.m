sup = zeros(231,195);

for i=1:length(directory_list)
  fprintf('\ngetting sup %s', ['../unpadded/' directory_list(i).name]);
  I = pgmRead(['../unpadded/' directory_list(i).name]);

  sup = sup + (I==255);
end

sup = sup/max(sup(:));
