%% Center the eyes

% eyelocs has the location of the eyes of face i:
%    eyelocs(i,:) = [x1 y1 x2 y2];
% and directory_list = dir('yalefaces/*.pgm');


eyex = round((eyelocs(:,1)+eyelocs(:,3))/2);
eyey = round((eyelocs(:,2)+eyelocs(:,4))/2)+top_pad;

for i=1:length(directory_list)
  fprintf('centering %s \n', ['../Data/padded/' directory_list(i).name]);
  I = pgmRead(['../Data/padded/' directory_list(i).name]);

  Io = I((-115:115)+eyey(i) , (-97:97)+eyex(i));

  pgmWrite(Io, ['../Data/centered/' directory_list(i).name], [0 255]);
end
