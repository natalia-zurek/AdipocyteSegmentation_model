function tiles = extract_tiles_from_image(I, tile_width, tile_height)

% Get the size of the image
    [image_height, image_width, ~] = size(I);

    % Calculate the number of tiles in the horizontal and vertical directions
    num_horizontal_tiles = floor(image_width / tile_width);
    num_vertical_tiles = floor(image_height / tile_height);

    % Initialize the cell array to store tiles
    tiles = cell(num_vertical_tiles* num_horizontal_tiles,1);

    % Extract non-overlapping tiles
    k = 1;
    for i = 1:num_vertical_tiles
        for j = 1:num_horizontal_tiles
            % Calculate the coordinates for each tile
            start_row = (i - 1) * tile_height + 1;
            end_row = i * tile_height;
            start_col = (j - 1) * tile_width + 1;
            end_col = j * tile_width;

            % Extract the tile from the image
            tiles{k} = I(start_row:end_row, start_col:end_col, :);
            k = k+1;
        end
    end

end