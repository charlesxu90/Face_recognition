
for i=1:length(directory_list)
  fprintf('unpadding %s \n', ['../Data/rotated/' directory_list(i).name]);
  I = pgmRead(['../Data/rotated/' directory_list(i).name]);

  center = round(size(I)/2);

  Io = I((-115:115)+center(1) , (-97:97)+center(2));
  Io = Io(1:2:size(Io,1), 1:2:size(Io,2));


  pgmWrite(Io, ['../Data/unpadded/' directory_list(i).name], [0 255]);
end
