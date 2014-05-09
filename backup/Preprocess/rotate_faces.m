%% Align the eyes by rotation
% Images(:,:,i) is a face i.
% eyelocs has the location of the eyes of face i:
%    eyelocs(i,:) = [x1 y1 x2 y2];
% and directory_list = dir('../faces/*.pgm');

% For each image, grab a 190(h) image.

dx = eyelocs(:,3)-eyelocs(:,1);
dy = eyelocs(:,2)-eyelocs(:,4);
rotation = atan(dy./dx) *180/pi;

for i=1:length(directory_list)
  fprintf('rotating ../Data/centered%s \n', directory_list(i).name);
%   ['pnmrotate ' num2str(-rotation(i)) ' ../centered/' ...
% 	 directory_list(i).name ' > ../rotated/' directory_list(i).name ];
  unix( ['pnmrotate ' num2str(-rotation(i)) ' ../Data/centered/' ...
	 directory_list(i).name ' > ../Data/rotated/' directory_list(i).name ] );
end
