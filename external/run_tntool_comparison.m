% Direct comparison with P. Koev's TNTool TNEigenValues routine.
%
% Prerequisite: download TNTool from
% https://sites.google.com/sjsu.edu/plamenkoev/home/software/tntool
% and add its directory to the MATLAB/Octave path before running this script.

this_file = mfilename('fullpath');
external_dir = fileparts(this_file);
root_dir = fileparts(external_dir);

if exist('TNEigenValues', 'file') ~= 2
    error(['TNEigenValues was not found. Add the official TNTool folder ', ...
           'to the MATLAB/Octave path, then run this script again.']);
end

bd_file = fullfile(root_dir, 'data', 'koev_tntool_bd.csv');
ref_file = fullfile(root_dir, 'data', 'koev_tntool_reference.csv');
out_file = fullfile(root_dir, 'results', 'koev_tntool_comparison.csv');

BD = dlmread(bd_file, ',');
reference_data = dlmread(ref_file, ',', 1, 0);
reference = reference_data(:, 2);

tic;
lambda = TNEigenValues(BD);
elapsed_seconds = toc;
lambda = sort(real(lambda(:)), 'descend');

if length(lambda) ~= length(reference)
    error('TNTool returned an unexpected number of eigenvalues.');
end

relative_error = abs(lambda - reference) ./ abs(reference);

fid = fopen(out_file, 'w');
if fid < 0
    error('Could not open the output CSV for writing.');
end
fprintf(fid, 'mode,independent_reference,koev_tntool,relative_error,elapsed_seconds\n');
for k = 1:length(lambda)
    fprintf(fid, '%d,%.17e,%.17e,%.17e,%.17e\n', ...
            k, reference(k), lambda(k), relative_error(k), elapsed_seconds);
end
fclose(fid);

fprintf('TNTool comparison completed.\n');
fprintf('Maximum relative error: %.6e\n', max(relative_error));
fprintf('Elapsed time: %.6e s\n', elapsed_seconds);
fprintf('Output: %s\n', out_file);
