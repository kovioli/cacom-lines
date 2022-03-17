%% OPEN AND SAVE DICOM DATA AS AN IMAGE FILE
    %Variables
        %filename = enclosed by 'filename', name of the dicom file to be converted
        %imagetype = enclose by 'imagetype', name of the output image type (png,bmp,jpg,png,tiff,gif etc..)
        %origin_folder = folder where the '.dcm' files are stored
        %target_folder = folder where the converted images (to '.png') are to be saved
        
function dicom2image(filename,imagetype, origin_folder, target_folder)
% Dicom to Image converter
% AUTHOR: Rance Tino
% VERSION: 0.1

%% Input Error Check
%%%%%%%%%%%%%%%%%%%%%%% input name %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~exist('filename','var') % doesn't exist
    error('You need an input name!');
end
%%%%%%%%%%%%%%%%%%%%%%% image type %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~exist('imagetype','var') % doesn't exist
    error('Please specify image type!');
end
%% Read and DICOM data
dcmfile = dicomread(fullfile(origin_folder, filename));

%% Show Original DICOM(.dcm) Image
%figure ()
%imshow(dcmfile,[]);
%title('Original DICOM Image');

%% Convert DICOM to image file
dcmImagei = uint8(255 * mat2gray(dcmfile)); %Convert to uint8 format
if(imagetype == 'png')
    imwrite(dcmImagei, fullfile(target_folder, strcat(filename,'.png')), imagetype)
    %imwrite(dcmImagei,'HU_dcmImage.png', imagetype);% Save Image to specified image type
    display('Finished saving .png image');
elseif(imagetype == 'bmp')
    imwrite(dcmImagei,'HU_dcmImage.bmp', imagetype);
     display('Finished saving .bmp image');
elseif(imagetype == 'jpg')
    imwrite(dcmImagei,'HU_dcmImage.jpg', imagetype);    
     display('Finished saving .jpg image');
elseif(imagetype == 'tif')
    imwrite(dcmImagei,'HU_dcmImage.tif', imagetype);  
     display('Finished saving .tif image');
elseif(imagetype == 'gif')
    imwrite(dcmImagei,'HU_dcmImage.gif', imagetype); 
    display('Finished saving .gif image');
else
    Display('Saving error');
end

end