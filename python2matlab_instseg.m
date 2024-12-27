function [inst_map, inst_types, inst_ids, inst_scores] = python2matlab_instseg(inst_map, varargin)

% Create input parser
parser = inputParser;
parser.CaseSensitive = true;

% Add parameters with validation functions
addRequired(parser, 'inst_map', @validateParam1);
addParameter(parser, 'inst_types', [], @validateParam2);
addParameter(parser, 'inst_ids', [], @validateParam2);
addParameter(parser, 'inst_scores', [], @validateParam2);
% Add additional parameters with corresponding validation functions

% Parse inputs
parse(parser, inst_map, varargin{:});

% Get parsed values
inst_map = parser.Results.inst_map;
inst_types = parser.Results.inst_types;
inst_ids = parser.Results.inst_ids;
inst_scores = parser.Results.inst_scores;

inst_types = inst_types+1;

zero_idx = inst_ids == 0;
if sum(zero_idx) == 0
    inst_map(inst_map == -1) = 0;
elseif sum(zero_idx) == 1
    new_id = inst_ids(end)+1;
    inst_map(inst_map == 0) = new_id;
    inst_map(inst_map == -1) = 0;
    inst_ids(zero_idx) = new_id;

else
    error('more than one zero index')

end

end

% Validation functions for parameters
function validateParam1(value)
if ~(isnumeric(value) && ismatrix(value))
    error('inst_map must be numerical matrix');
end
end

function validateParam2(value)
if ~(isnumeric(value) && isvector(value))
    error('params must be numerical vectors');
end
end