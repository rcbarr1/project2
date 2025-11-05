%%%
% test to compare matlab sparse solve with scipy and petsc
% see exp20/exp21/exp21_speedtest python files
%%%

%% load ocim
addpath '/Users/Reese_1/Documents/Research Projects/project2/examples/run_DAC_OAE/jonathansharp-CO2-System-Extd-3ae63d5/main'

% load model
fprintf('loading model...\n')

data_path = '/Users/Reese_1/Documents/Research Projects/project2/data/';
load(strcat(data_path, 'OCIM2_48L_base/OCIM2_48L_base_transport.mat'));
M3d = ncread(strcat(data_path, 'OCIM2_48L_base/OCIM2_48L_base_data.nc'), 'ocnmask');
VOL = ncread(strcat(data_path, 'OCIM2_48L_base/OCIM2_48L_base_data.nc'), 'vol');
model_lon = ncread(strcat(data_path, 'OCIM2_48L_base/OCIM2_48L_base_data.nc'), 'tlon');
model_lon = transpose(squeeze(model_lon(1, :, 1)));
model_lat = ncread(strcat(data_path, 'OCIM2_48L_base/OCIM2_48L_base_data.nc'), 'tlat');
model_lat = squeeze(model_lat(:, 1, 1));
model_depth = ncread(strcat(data_path, 'OCIM2_48L_base/OCIM2_48L_base_data.nc'), 'tz');
model_depth = squeeze(model_depth(1, 1, :));
iocn = find(M3d==1);
isurf = find(M3d(:, :, 1) == 1);
m = length(iocn);
ns = length(isurf);
[ny,nx,nz] = size(M3d);
Vocn = VOL(iocn);

%% load in sparse matrix from python
load('./LHS_1hr.mat')
load('./RHS_1hr.mat')

%for i = 1:4
%tic
%DM = decomposition(LHS);
%toc

tic
x = LHS\RHS.';
toc
%end

