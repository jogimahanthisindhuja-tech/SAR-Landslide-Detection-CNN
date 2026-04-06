
clc; clear; close all;

%% =========================================
% STEP 1 — LOAD & CLEAN InSAR DATA
% =========================================
data = readmatrix('Asc-InSAR.txt');

x = data(:,1);
y = data(:,2);
d = data(:,3);

% Remove NaN / Inf
valid = isfinite(x) & isfinite(y) & isfinite(d);
x = x(valid);
y = y(valid);
d = d(valid);

% Remove extreme outliers
d(abs(d) > 200) = NaN;
valid = isfinite(d);
x = x(valid);
y = y(valid);
d = d(valid);

disp(['Valid points: ', num2str(length(x))]);

%% =========================================
% STEP 2 — GRID INTERPOLATION
% =========================================
xq = linspace(min(x), max(x), 200);
yq = linspace(min(y), max(y), 200);
[Xq, Yq] = meshgrid(xq, yq);

Z = griddata(x, y, d, Xq, Yq, 'natural');

Z(isnan(Z)) = 0;

% Normalize
Z = (Z - min(Z(:))) / (max(Z(:)) - min(Z(:)));

figure;
imagesc(xq, yq, Z); axis xy;
colorbar; colormap jet;
title('Normalized InSAR Deformation');

%% =========================================
% STEP 3 — LOAD HFSS RADIATION PATTERN
% =========================================
gain_data = readmatrix('gain_table.csv');

phi   = deg2rad(gain_data(:,2));
theta = deg2rad(gain_data(:,3));
G_dB  = gain_data(:,4);

G_lin = 10.^(G_dB/10);

phi_vals   = unique(phi);
theta_vals = unique(theta);

[Phi_grid, Theta_grid] = meshgrid(phi_vals, theta_vals);
G_grid = zeros(length(theta_vals), length(phi_vals));

for i = 1:length(phi)
    t_idx = find(theta_vals == theta(i));
    p_idx = find(phi_vals   == phi(i));
    G_grid(t_idx, p_idx) = G_lin(i);
end

%% =========================================
% STEP 4 — MAP RADIATION GAIN
% =========================================
radar_pos = [0, 0, 50];
G_map = zeros(size(Z));

for i = 1:size(Z,1)
    for j = 1:size(Z,2)
        
        dx = Xq(i,j) - radar_pos(1);
        dy = Yq(i,j) - radar_pos(2);
        dz = -radar_pos(3);
        
        R = sqrt(dx^2 + dy^2 + dz^2);
        
        theta_t = acos(dz / R);
        phi_t   = atan2(dy, dx);
        
        if phi_t < 0
            phi_t = phi_t + 2*pi;
        end
        
        G_map(i,j) = interp2(Phi_grid, Theta_grid, G_grid, phi_t, theta_t, 'linear', 0);
    end
end

% Normalize gain
G_map = (G_map - min(G_map(:))) / (max(G_map(:)) - min(G_map(:)));

figure;
imagesc(G_map); colorbar; colormap jet;
title('Radiation Gain Map');

%% =========================================
% STEP 5 — APPLY RADIATION WEIGHTING
% =========================================
Z_weighted = Z .* G_map;

Z_weighted = (Z_weighted - min(Z_weighted(:))) / (max(Z_weighted(:)) - min(Z_weighted(:)));

figure;
imagesc(Z_weighted); axis xy;
colorbar; colormap jet;
title('Radiation-Aware Deformation');

%% =========================================
% STEP 6 — PATCH CREATION
% =========================================
patch_size = 16;
stride = 8;

patches = {};
labels = [];

threshold = 0.6;   % adjust if needed
mask = Z_weighted > threshold;

count = 1;

for i = 1:stride:200-patch_size
    for j = 1:stride:200-patch_size
        
        patch = Z_weighted(i:i+patch_size-1, j:j+patch_size-1);
        patch_mask = mask(i:i+patch_size-1, j:j+patch_size-1);
        
        patches{count} = patch;
        
        if sum(patch_mask(:)) > 10
            labels(count) = 1;
        else
            labels(count) = 0;
        end
        
        count = count + 1;
    end
end

%% =========================================
% STEP 7 — PREPARE CNN DATA
% =========================================
num_samples = length(patches);

X = zeros(16,16,1,num_samples);

for i = 1:num_samples
    X(:,:,1,i) = patches{i};
end

Y = categorical(labels);

%% =========================================
% STEP 8 — CNN MODEL
% =========================================
layers = [
    imageInputLayer([16 16 1])

    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(32)
    reluLayer

    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs',10, ...
    'MiniBatchSize',32, ...
    'Plots','training-progress', ...
    'Verbose',false);

net = trainNetwork(X, Y, layers, options);

%% =========================================
% STEP 9 — EVALUATION
% =========================================
YPred = classify(net, X);

accuracy = sum(YPred == Y) / numel(Y);
disp(['Accuracy: ', num2str(accuracy*100), '%']);

figure;
confusionchart(Y, YPred);
title('Confusion Matrix');

%% =========================================
% STEP 10 — PREDICTION MAP
% =========================================
prediction_map = zeros(size(Z_weighted));

for i = 1:stride:200-patch_size
    for j = 1:stride:200-patch_size
        
        patch = Z_weighted(i:i+patch_size-1, j:j+patch_size-1);
        patch = reshape(patch, [16 16 1 1]);
        
        pred = classify(net, patch);
        
        if pred == '1'
            prediction_map(i:i+patch_size-1, j:j+patch_size-1) = 1;
        end
    end
end

figure;
imagesc(prediction_map);
colormap hot;
title('CNN Landslide Detection');

%% =========================================
% STEP 11 — FINAL OVERLAY (BEST OUTPUT)
% =========================================
figure;
imagesc(xq, yq, Z_weighted);
axis xy; hold on;

contour(xq, yq, prediction_map, [1 1], 'r', 'LineWidth', 2);

colorbar;
colormap jet;
title('Final Landslide Detection (Radiation-Aware CNN)');
