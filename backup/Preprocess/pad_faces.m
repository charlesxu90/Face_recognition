%% Squares up the faces. this is good useful for rotating them

Io = ones(320,320)*255;
top_pad = round((320-243)/2);

for i=1:length(directory_list)
  fprintf('padding yalefaces/%s \n', directory_list(i).name);
  I=pgmRead(['../Data/Original/' directory_list(i).name]);

  Io(top_pad+(0:size(I,1)-1), 1:size(I,2)) = I;

  pgmWrite(Io, ['../Data/padded/' directory_list(i).name]);
end
