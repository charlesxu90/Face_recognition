%% This file loads global variables

olddir = pwd; % Save current folder
chdir('../Data/Original'); % Change to faces folder

directory_list = dir('*.pgm'); % Read directory list
% get rid of . and ..
directory_list = directory_list(3:length(directory_list));

chdir(olddir);


load ../Data/eyelocs.mat -ascii eyelocs
% What is this used for????
support = pgmRead('../Data/support.pgm')>0;
