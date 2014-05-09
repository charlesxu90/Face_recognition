directory_list = dir('../faces/*.pgm');
nfiles = length(directory_list);
axis ij;

I = pgmRead(['../faces/' directory_list(i).name]);
Images = zeros(size(I,1), size(I,2), nfiles);

for i=1:nfiles
fprintf('\nprefetching %s', ['../faces/' directory_list(i).name]);
  Images(:,:,i) = pgmRead(['../faces/' directory_list(i).name])/255;
end

for i=1:nfiles
  fprintf('\nshowing %s', ['../faces/' directory_list(i).name]);

  clf
  imagesc(Images(:,:,i));
  hold on;

  [x1, y1] = ginput(1);
  plot(x1,y1, 'rx');
  [x2, y2] = ginput(1);
  plot(x2,y2, 'bx');
  drawnow
  eyelocs(i,:) = [x1 y1 x2 y2];

  hold off;
pause;
end


eyedistances = sqrt((eyelocs(:,1)- eyelocs(:,3)).^2 + ...
                    (eyelocs(:,2)- eyelocs(:,4)).^2);
save -ascii eyedistances.dat eyedistances
