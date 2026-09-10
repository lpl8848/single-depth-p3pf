function export_tum_od(dataset_path, output_file, max_pairs, frame_gap)
%EXPORT_TUM_OD Export correctly associated TUM RGB-D frame pairs.
% Each correspondence stores a reference-camera 3D point, a centered query
% pixel, the real query-frame z-depth, and the descriptor match metric.

if nargin < 3, max_pairs = inf; end
if nargin < 4, frame_gap = 10; end

[rgb_ts, rgb_files] = read_timed_list(fullfile(dataset_path, 'rgb.txt'));
[depth_ts, depth_files] = read_timed_list(fullfile(dataset_path, 'depth.txt'));
[gt_ts, gt_t, gt_q] = read_groundtruth(fullfile(dataset_path, 'groundtruth.txt'));

% TUM Freiburg calibration.  The experiment keeps principal point and pixel
% aspect ratio known, while hiding the common focal scale from OD-P3Pf.
if contains(lower(dataset_path), 'freiburg1')
    fx = 517.3; fy = 516.5; cx = 318.6; cy = 255.3;
elseif contains(lower(dataset_path), 'freiburg2')
    fx = 520.9; fy = 521.0; cx = 325.1; cy = 249.7;
else
    error('Unknown TUM calibration family: %s', dataset_path);
end
depth_factor = 5000.0;
max_time_gap = 0.03;

out_dir = fileparts(output_file);
if ~isempty(out_dir) && ~exist(out_dir, 'dir'), mkdir(out_dir); end
fid = fopen(output_file, 'w');
if fid < 0, error('Cannot open output: %s', output_file); end
cleanup = onCleanup(@() fclose(fid));

fprintf(fid, '# TUM_OD_P3PF_EXPORT_V1\n');
fprintf(fid, '# correspondence: X Y Z centered_u isotropic_v query_z match_metric\n');
fprintf(fid, '# focal convention: f=fx, isotropic_v=(fx/fy)*(v-cy)\n');

written = 0;
candidate_ids = 1:frame_gap:(numel(rgb_ts) - frame_gap);
for list_index = 1:numel(candidate_ids)
    if written >= max_pairs, break; end
    i = candidate_ids(list_index);
    j = i + frame_gap;

    [ref_depth_gap, ref_depth_id] = min(abs(depth_ts - rgb_ts(i)));
    [qry_depth_gap, qry_depth_id] = min(abs(depth_ts - rgb_ts(j)));
    [ref_gt_gap, ref_gt_id] = min(abs(gt_ts - rgb_ts(i)));
    [qry_gt_gap, qry_gt_id] = min(abs(gt_ts - rgb_ts(j)));
    if max([ref_depth_gap, qry_depth_gap, ref_gt_gap, qry_gt_gap]) > max_time_gap
        continue;
    end

    image1 = imread(fullfile(dataset_path, rgb_files{i}));
    image2 = imread(fullfile(dataset_path, rgb_files{j}));
    if size(image1, 3) == 3, gray1 = rgb2gray(image1); else, gray1 = image1; end
    if size(image2, 3) == 3, gray2 = rgb2gray(image2); else, gray2 = image2; end
    depth1 = double(imread(fullfile(dataset_path, depth_files{ref_depth_id}))) / depth_factor;
    depth2 = double(imread(fullfile(dataset_path, depth_files{qry_depth_id}))) / depth_factor;

    points1 = detectSURFFeatures(gray1, 'MetricThreshold', 100);
    points2 = detectSURFFeatures(gray2, 'MetricThreshold', 100);
    [features1, valid1] = extractFeatures(gray1, points1);
    [features2, valid2] = extractFeatures(gray2, points2);
    [pairs, metric] = matchFeatures(features1, features2, ...
        'MaxRatio', 0.75, 'Unique', true);
    if size(pairs, 1) < 6, continue; end

    location1 = valid1(pairs(:, 1)).Location;
    location2 = valid2(pairs(:, 2)).Location;
    col1 = round(location1(:, 1)); row1 = round(location1(:, 2));
    col2 = round(location2(:, 1)); row2 = round(location2(:, 2));
    inside = col1 >= 1 & col1 <= size(depth1, 2) & ...
             row1 >= 1 & row1 <= size(depth1, 1) & ...
             col2 >= 1 & col2 <= size(depth2, 2) & ...
             row2 >= 1 & row2 <= size(depth2, 1);
    location1 = location1(inside, :); location2 = location2(inside, :);
    col1 = col1(inside); row1 = row1(inside);
    col2 = col2(inside); row2 = row2(inside);
    metric = metric(inside);
    if isempty(metric), continue; end

    z1 = depth1(sub2ind(size(depth1), row1, col1));
    z2 = depth2(sub2ind(size(depth2), row2, col2));
    valid_depth = z1 > 0.1 & z1 < 5.0 & z2 > 0.1 & z2 < 5.0;
    location1 = location1(valid_depth, :);
    location2 = location2(valid_depth, :);
    z1 = z1(valid_depth); z2 = z2(valid_depth); metric = metric(valid_depth);
    if numel(z1) < 6, continue; end

    % MATLAB image locations are one-based; TUM intrinsics use zero-based
    % pixel coordinates.
    u1 = location1(:, 1) - 1.0; v1 = location1(:, 2) - 1.0;
    u2 = location2(:, 1) - 1.0; v2 = location2(:, 2) - 1.0;
    world = [(u1 - cx) .* z1 / fx, (v1 - cy) .* z1 / fy, z1];
    centered_u = u2 - cx;
    isotropic_v = (fx / fy) * (v2 - cy);

    [metric, order] = sort(metric, 'ascend');
    world = world(order, :); centered_u = centered_u(order);
    isotropic_v = isotropic_v(order); z2 = z2(order);

    R1 = quat2rotm([gt_q(ref_gt_id, 4), gt_q(ref_gt_id, 1:3)]);
    R2 = quat2rotm([gt_q(qry_gt_id, 4), gt_q(qry_gt_id, 1:3)]);
    t1 = gt_t(ref_gt_id, :)'; t2 = gt_t(qry_gt_id, :)';
    R_rel = R2' * R1;
    t_rel = R2' * (t1 - t2);

    written = written + 1;
    fprintf(fid, 'FRAME %d %d %.9f %.9f\n', written, size(world, 1), rgb_ts(i), rgb_ts(j));
    fprintf(fid, 'INTRINSICS %.12g %.12g %.12g %.12g\n', fx, fy, cx, cy);
    pose = [R_rel, t_rel];
    fprintf(fid, 'POSE ');
    fprintf(fid, '%.12g ', pose');
    fprintf(fid, '\n');
    data = [world, centered_u, isotropic_v, z2, metric];
    fprintf(fid, '%.12g %.12g %.12g %.12g %.12g %.12g %.12g\n', data');

    fprintf('exported pair %d: rgb %d -> %d, %d matches\n', ...
        written, i, j, size(world, 1));
end

fprintf('Wrote %d frame pairs to %s\n', written, output_file);
end

function [timestamps, files] = read_timed_list(path)
fid = fopen(path, 'r');
if fid < 0, error('Cannot open %s', path); end
cleanup = onCleanup(@() fclose(fid));
timestamps = []; files = {};
while true
    line = fgetl(fid);
    if ~ischar(line), break; end
    line = strtrim(line);
    if isempty(line) || line(1) == '#', continue; end
    fields = strsplit(line);
    timestamps(end + 1, 1) = str2double(fields{1}); %#ok<AGROW>
    files{end + 1, 1} = fields{2}; %#ok<AGROW>
end
end

function [timestamps, translation, quaternion] = read_groundtruth(path)
fid = fopen(path, 'r');
if fid < 0, error('Cannot open %s', path); end
cleanup = onCleanup(@() fclose(fid));
rows = [];
while true
    line = fgetl(fid);
    if ~ischar(line), break; end
    line = strtrim(line);
    if isempty(line) || line(1) == '#', continue; end
    values = sscanf(line, '%f')';
    if numel(values) == 8, rows(end + 1, :) = values; end %#ok<AGROW>
end
timestamps = rows(:, 1);
translation = rows(:, 2:4);
quaternion = rows(:, 5:8);
end
