% Expanded directory paths stored in a cell array
expandedDirs = {
    '/Users/peteruba/Desktop/image processing and data visualization/HW4/expanded dataset/Cells_1to25',
    '/Users/peteruba/Desktop/image processing and data visualization/HW4/expanded dataset/Cells_26to49',
    '/Users/peteruba/Desktop/image processing and data visualization/HW4/expanded dataset/Cells_51to81'
};

% Process each expanded directory
for j = 1:length(expandedDirs)
    % Get a list of TIFF files in the current directory
    files = [dir(fullfile(expandedDirs{j}, '*.tif')); dir(fullfile(expandedDirs{j}, '*.TIF'))];
    
    fprintf('Processing directory: %s\n', expandedDirs{j});
    fprintf('Number of files found: %d\n', length(files));
    
    for i = 1:length(files)
        fullPath = fullfile(files(i).folder, files(i).name);
        
        try
            % Read the image
            img = imread(fullPath);
            
            % Perform flips, rotations, translations
            flippedUD = flipud(img);
            rotated90 = imrotate(img, 90);
            translatedImg = imtranslate(img, [10, 5]);  % Make sure your MATLAB version supports this syntax
            
            % Save each version in the same directory
            imwrite(flippedUD, fullfile(files(i).folder, ['Flip_' files(i).name]));
            imwrite(rotated90, fullfile(files(i).folder, ['R90_' files(i).name]));
            imwrite(translatedImg, fullfile(files(i).folder, ['Trans_' files(i).name]));
            
        catch ME
            warning('Failed to process %s due to error: %s\n', fullPath, ME.message);
        end
    end
end
