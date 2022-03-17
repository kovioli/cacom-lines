%convert whole batch
origin_folder = 'data_dcm'; %folder where the "Document" type .dcm files are to be found
target_folder = 'data_png'; %folder where the converted images are saved

if ~exist(target_folder, 'dir')
   mkdir(target_folder)
end

filelist = dir(fullfile(origin_folder));

n = 1;
while n < length(filelist)+1
    if filelist(n).isdir | strcmp(filelist(n).name, '.DS_Store')
         %== '.DS_Store'
        filelist(n) = [];
    else
        n = n+1;
    end
end


cell_filelist = struct2cell(filelist);

file_names = cell_filelist(1, :);

for f = 1:length(file_names)
    %k = char(file_names(f))
    %strcat(k(1:end-4),'.png')
    dicom2image(char(file_names(f)), 'png', origin_folder, target_folder);
end


