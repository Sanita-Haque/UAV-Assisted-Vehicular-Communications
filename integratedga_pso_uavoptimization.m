% Define parameters for simulation
U = 2;  % Number of UAVs
V = 3;  % Number of vehicles
K = 10;  % Number of subchannels
N = 20;  % Number of time slots
xi = 0.7;  % Weighting factor for throughput and energy efficiency
P_max = 11;  % Maximum transmit power
E_max = 1.5e5;  % Maximum energy required for communication
S_min = 25;  % Minimum speed
S_max = 35;  % Maximum speed
delta_t = 0.01;  % Time step
d_min = 10;  % Minimum distance to avoid collision
H = 100;  % UAV height
W = 20;  % Bandwidth
sigma = 10e-14;  % Noise power
h0 = 1e-4;  % Channel gain at reference distance
length_of_highway = 1000;
lanes = 1;
mu = 65;
sigma_v = 5;

% Parameters for UAV flying power consumption
P0 = 79.86;
P_i = 88.63;
u_tip = 120;
s0 = 4.03;
e0 = 0.6;
Delta = 0.012;
theta_rotor = 300;
R = 0.4;
B = 0.503;
iota = 1.225;
C = 20;
l = 0.1;

% Initial conditions
q_x_init = linspace(0, length_of_highway, U); % Initialize all UAVs' x-coordinates to 0
q_y_init = 100 * ones(1, U); % Constant altitude of 100m for all UAVs
S_x_init = linspace(S_min, S_max, U); % Set initial speeds between min and max speeds

% Generate initial vehicle positions based on Poisson distribution
x_v_init = vehicle_model_wrap(V, N, delta_t, mu, sigma_v, lanes, length_of_highway);
y_v_init = zeros(lanes, V, N); % y-positions of vehicles

% Decision variables
P = optimvar('P', V, U, K, N);
q_x = optimvar('q_x', U, N);
q_y = optimvar('q_y', U, N, 'LowerBound', 100, 'UpperBound', 100); % Constant altitude
S_x = optimvar('S_x', U, N, 'LowerBound', S_min, 'UpperBound', S_max);
phi_x = optimvar('phi_x', U, N-1, 'LowerBound', -150, 'UpperBound', 10);

omega = optimvar('omega', V, U, N, 'Type', 'integer', 'LowerBound', 0, 'UpperBound', 1);
alpha = optimvar('alpha', V, K, N, 'Type', 'integer', 'LowerBound', 0, 'UpperBound', 1);

% Objective function
obj = fcn2optimexpr(@(q_x, q_y, P, omega, alpha, S_x, phi_x) ...
    ObjectiveFunction(q_x, q_y, P, omega, alpha, S_x, phi_x, ...
    x_v_init, y_v_init, xi, W, sigma, h0, U, V, K, N, H, P0, P_i, u_tip, s0, e0, iota, B, delta_t), ...
    q_x, q_y, P, omega, alpha, S_x, phi_x, 'OutputSize', [1, 1]);

% Optimization problem definition
prob = optimproblem('Objective', obj);

% Add constraints individually
prob.Constraints.userAssocCons = userAssociationConstraint(omega, V, N, U);
prob.Constraints.subchannelAssignCons = subchannelAssignmentConstraint(alpha, V, K, N);
prob.Constraints.antiCollisionCons = antiCollisionConstraint(q_x, q_y, d_min, U, N);
prob.Constraints.positionUpdateCons = positionUpdateConstraint(q_x, q_x_init, S_x, phi_x, delta_t, U, N);
[c_update, c_lower, c_upper] = speedUpdateConstraint(S_x, S_x_init, phi_x, delta_t, U, N, S_min, S_max);
prob.Constraints.speedUpdateCons = c_update;
prob.Constraints.speedLowerCons = c_lower;
prob.Constraints.speedUpperCons = c_upper;
prob.Constraints.energyLowerCons = energyLowerConstraint(P, S_x, delta_t, P0, P_i, u_tip, s0, e0, iota, B, U, N);
prob.Constraints.energyUpperCons = energyUpperConstraint(P, S_x, delta_t, P0, P_i, u_tip, s0, e0, iota, B, E_max, U, N);
prob.Constraints.transmitPowerUpperCons = transmitPowerConstraint(P, P_max, V, U, K, N);

% Initial guess
x0.q_x = repmat(q_x_init', 1, N);
x0.q_y = repmat(q_y_init', 1, N);
% Exponential increase in speed
t = linspace(0, 1, N);
x0.S_x = repmat(S_min + (S_max - S_min) * (1 - exp(-5 * t)), U, 1);
x0.phi_x = 0.1 * rand(U, N-1) - 0.05; % Small random values centered around 0
x0.omega = randi([0, 1], V, U, N);
x0.alpha = randi([0, 1], V, K, N);
x0.P = P_max * rand(V, U, K, N); % Initial power allocation using only P_max


% Solve the problem using genetic algorithm (ga) with parallel computing
optionsGA = optimoptions('ga', ...
    'Display', 'iter', ...
    'MaxGenerations', 50, ...
    'PopulationSize', 50, ...
    'CrossoverFraction', 0.8, ...
    'MutationFcn', @mutationuniform, ...
    'UseParallel', true, ...
    'HybridFcn', []); % Disable hybrid function for initial GA run

% Start parallel pool if not already started
if isempty(gcp('nocreate'))
    parpool;
end

[solGA, fvalGA] = solve(prob, x0, 'Solver', 'ga', 'Options', optionsGA);

% Convert GA solution to initial guess for particleswarm
x0PSO = struct2array(solGA);

% Define lower and upper bounds for particleswarm
lb = zeros(size(x0PSO));
ub = ones(size(x0PSO)) * P_max; % Upper bound for power allocation

% Solve the problem using particleswarm (PSO) with the initial guess from GA
optionsPSO = optimoptions('particleswarm', ...
    'Display', 'iter', ...
    'SwarmSize', 50, ...
    'HybridFcn', @patternsearch, ...
    'UseParallel', true);

% Flatten objective function for particleswarm
objectivePSO = @(x) ObjectiveFunctionPSO(x, x_v_init, y_v_init, xi, W, sigma, h0, U, V, K, N, H, P0, P_i, u_tip, s0, e0, iota, B, delta_t, solGA);

% Solve the problem using particleswarm (PSO)
[solPSO, fvalPSO] = particleswarm(objectivePSO, length(x0PSO), lb, ub, optionsPSO);

% Convert particleswarm solution back to struct
solPSO = array2struct(solPSO, solGA);

% Output of the final solution
disp('Optimized positions (q_x):');
disp(solPSO.q_x);

disp('Optimized positions (q_y):');
disp(solPSO.q_y);

disp('Optimized speeds (S_x):');
disp(solPSO.S_x);

disp('Optimized accelerations (phi_x):');
disp(solPSO.phi_x);

disp('Optimized transmit power (P):');
disp(solPSO.P);

disp('Optimized user associations (omega):');
disp(solPSO.omega);

disp('Optimized subchannel allocations (alpha):');
disp(solPSO.alpha);

% Objective function value
disp('Objective function value:');
disp(fvalPSO);

% Visualization
% Plot Initial and Optimized UAV Speeds Over Time
figure;
hold on;

% Plot initial UAV speeds
for u = 1:U
    plot(1:N, x0.S_x(u, :), '--o', 'DisplayName', ['Initial UAV ' num2str(u)]);
end

% Plot optimized UAV speeds
for u = 1:U
    plot(1:N, solPSO.S_x(u, :), '-x', 'DisplayName', ['Optimized UAV ' num2str(u)]);
end

xlabel('Time Slot');
ylabel('Speed');
title('Initial and Optimized UAV Speeds Over Time');
legend('show');
grid on;
hold off;

omega_init = x0.omega;
alpha_init = x0.alpha;
P_init = x0.P;
% Calculate total data rate and energy consumption before optimization
total_data_rate_before = zeros(1, N);
total_energy_before = zeros(1, N);
data_rate_before = zeros(U, N, V, K); % Store data rate for each u, n, v, k before optimization

for n = 1:N
    for u = 1:U
        for v = 1:V
            for k = 1:K
                d_uv = sqrt((q_x_init(u) - x_v_init(1, v, n))^2 + (q_y_init(u) - 0)^2 + H^2);
                h_uv = h0 / (d_uv^2);
                snr_uv = h_uv / sigma^2;
                data_rate_uvkn = W * log2(1 + snr_uv * x0.P(v, u, k, n)); % Data rate before optimization
                data_rate_before(u, n, v, k) = data_rate_uvkn; % Store data rate
                total_data_rate_before(n) = total_data_rate_before(n) + data_rate_uvkn * x0.omega(v, u, n) * x0.alpha(v, k, n);
            end
        end
        S = S_x_init(u);
        flying_power_u_n = P0 * (1 + 3 * S^2 / u_tip^2) + P_i * sqrt(1 + S^4 / (4 * s0^4) - S^2 / (2 * s0^2)) + 0.5 * e0 * iota * B * S^3;
        total_energy_before(n) = total_energy_before(n) + sum(sum(x0.P(:, u, :, n), 3), 1) * delta_t + flying_power_u_n * delta_t;
    end
end

% Calculate total data rate and energy consumption after optimization
total_data_rate_after = zeros(1, N);
total_energy_after = zeros(1, N);
data_rate_after = zeros(U, N, V, K); % Store data rate for each u, n, v, k after optimization

for n = 1:N
    for u = 1:U
        for v = 1:V
            for k = 1:K
                d_uv = sqrt((solPSO.q_x(u, n) - x_v_init(1, v, n))^2 + (solPSO.q_y(u, n) - 0)^2 + H^2);
                h_uv = h0 / (d_uv^2);
                snr_uv = h_uv / sigma^2;
                data_rate_uvkn = W * log2(1 + snr_uv * solPSO.P(v, u, k, n)); % Data rate after optimization
                data_rate_after(u, n, v, k) = data_rate_uvkn; % Store data rate
                total_data_rate_after(n) = total_data_rate_after(n) + data_rate_uvkn * solPSO.omega(v, u, n) * solPSO.alpha(v, k, n);
            end
        end
        S = solPSO.S_x(u, n);
        flying_power_u_n = P0 * (1 + 3 * S^2 / u_tip^2) + P_i * sqrt(1 + S^4 / (4 * s0^4) - S^2 / (2 * s0^2)) + 0.5 * e0 * iota * B * S^3;
        total_energy_after(n) = total_energy_after(n) + sum(sum(solPSO.P(:, u, :, n), 3), 1) * delta_t + flying_power_u_n * delta_t;
    end
end

% Plot total data rate and energy consumption before and after optimization
figure;
plot(1:N, total_data_rate_before/1e2, '-o', 'DisplayName', 'Data Rate Pre Optimization');
hold on;
plot(1:N, total_data_rate_after/1e2, '-x', 'DisplayName', 'Data Rate Post Optimization');
xlabel('Time Slot(s)');
ylabel('Total Data Rate (Mbits)');
legend('show');
title('Total Data Rate Before and After Optimization');
yticks = get(gca, 'YTick');
set(gca, 'YTickLabel', yticks);
%ylim([0 800]);
hold off;

figure;
plot(1:N, total_energy_before, '-o', 'DisplayName', 'Energy Consumption Pre Optimization');
hold on;
plot(1:N, total_energy_after, '-x', 'DisplayName', 'Energy Consumption Post Optimization');
xlabel('Time Slot');
ylabel('Total Energy Consumption (W)');
legend('show');
title('Total Energy Consumption Before and After Optimization');
hold off;

% Helper function to flatten struct into array
function arr = struct2array(sol)
    fields = fieldnames(sol);
    totalElements = sum(structfun(@numel, sol));
    arr = zeros(totalElements, 1); % Preallocate the array
    
    startIdx = 1;
    for i = 1:length(fields)
        numElements = numel(sol.(fields{i}));
        arr(startIdx:startIdx + numElements - 1) = sol.(fields{i})(:);
        startIdx = startIdx + numElements;
    end
end

% Helper function to convert array back to struct
function solStruct = array2struct(arr, template)
    fields = fieldnames(template);
    solStruct = struct();
    startIdx = 1;
    for i = 1:length(fields)
        numElements = numel(template.(fields{i}));
        solStruct.(fields{i}) = reshape(arr(startIdx:startIdx + numElements - 1), size(template.(fields{i})));
        startIdx = startIdx + numElements;
    end
end

% Objective function for PSO
function f = ObjectiveFunctionPSO(x, x_v_init, y_v_init, xi, W, sigma, h0, U, V, K, N, H, P0, P_i, u_tip, s0, e0, iota, B, delta_t, template)
    xStruct = array2struct(x, template);
    % Evaluate the objective function using the original problem structure
    f = ObjectiveFunction(xStruct.q_x, xStruct.q_y, xStruct.P, xStruct.omega, xStruct.alpha, xStruct.S_x, xStruct.phi_x, ...
        x_v_init, y_v_init, xi, W, sigma, h0, U, V, K, N, H, P0, P_i, u_tip, s0, e0, iota, B, delta_t);
end

% Objective function definition
function f = ObjectiveFunction(q_x, q_y, P, omega, alpha, S_x, phi_x, x_v_init, y_v_init, xi, W, sigma, h0, U, V, K, N, H, P0, P_i, u_tip, s0, e0, iota, B, delta_t)
    total_data_rate = 0;
    total_energy = 0;
    regularization = 0;

    for n = 1:N
        % Data rate calculation
        for u = 1:U
            data_rate_u_n = 0;
            for v = 1:V
                for k = 1:K
                    % Distance calculation (dimension handling)
                    d_uv = sqrt((q_x(u, n) - x_v_init(1, v, n))^2 + (q_y(u, n) - y_v_init(1, v, n))^2 + H^2);
                    % Channel power gain
                    h_uv = h0 / (d_uv^2);
                    % Signal-to-noise ratio (SNR)
                    snr_uv = h_uv / sigma^2;
                    % Data rate calculation
                    data_rate_uvkn = W * log2(1 + snr_uv * P(v, u, k, n));
                    % Weighted data rate accumulation
                    data_rate_u_n = data_rate_u_n + omega(v, u, n) * alpha(v, k, n) * data_rate_uvkn;
                end
            end
            % Accumulate total data rate
            total_data_rate = total_data_rate + data_rate_u_n;

            % Energy consumption calculation
            S = S_x(u, n);
            % Flying power calculation
            flying_power_u_n = P0 * (1 + 3 * S^2 / u_tip^2) + P_i * sqrt(1 + S^4 / (4 * s0^4) - S^2 / (2 * s0^2)) + 0.5 * e0 * iota * B * S^3;
            % Communication energy calculation
            comm_energy_u_n = sum(P(:, u, :, n), 'all') * delta_t; % Adjust if needed based on P
            % Flight energy calculation
            flight_energy_u_n = flying_power_u_n * delta_t;
            % Total energy for this UAV at this time slot
            energy_u_n = comm_energy_u_n + flight_energy_u_n;
            % Accumulate total energy
            total_energy = total_energy + energy_u_n;

            % Regularization term to penalize extreme acceleration/deceleration
            if n > 1
                regularization = regularization + abs(phi_x(u, n-1));
            end
        end
    end

    % Objective function calculation
    f = xi * total_data_rate - (1 - xi) * total_energy - 0.01 * regularization; % Adjust the regularization coefficient as needed
end

% User association constraint definition
function c = userAssociationConstraint(omega, V, N, U)
    c = optimconstr(V, N);
    for v = 1:V
        for n = 1:N
            c(v, n) = sum(omega(v, 1:U, n)) == 1;
        end
    end
end

% Subchannel assignment constraint definition
function c = subchannelAssignmentConstraint(alpha, ~, K, N)
    % Initialize the constraints matrix
    c = optimconstr(K, N);

    % Loop through each k and n to set the constraints
    for k = 1:K
        for n = 1:N
            % Ensure at least one alpha(v, k, n) is 1 for each k and n
            c(k, n) = sum(alpha(:, k, n)) >= 1;
        end
    end
end

% Anti-collision constraint definition
function c = antiCollisionConstraint(q_x, q_y, d_min, U, N)
    c = optimconstr(U, U, N);
    for u = 1:U
        for j = 1:U
            if u ~= j
                for n = 1:N
                    c(u, j, n) = (q_x(u, n) - q_x(j, n))^2 + (q_y(u, n) - q_y(j, n))^2 >= d_min^2;
                end
            end
        end
    end
end

% Position update constraint definition
function c = positionUpdateConstraint(q_x, q_x_init, S_x, phi_x, delta_t, U, N)
    c = optimconstr(U, N);
    for u = 1:U
        for n = 1:N
            if n == 1
                c(u, n) = q_x(u, n) == q_x_init(u);
            else
                c(u, n) = q_x(u, n) == q_x(u, n-1) + S_x(u, n-1) * delta_t + 0.5 * phi_x(u, n-1) * delta_t^2;
            end
        end
    end
end

% Speed update constraint definition
function [c_update, c_lower, c_upper] = speedUpdateConstraint(S_x, S_x_init, phi_x, delta_t, U, N, S_min, S_max)
    c_update = optimconstr(U, N); % Initialize the speed update constraint matrix
    c_lower = optimconstr(U, N);  % Initialize the lower bound constraint matrix
    c_upper = optimconstr(U, N);  % Initialize the upper bound constraint matrix
    
    for u = 1:U
        for n = 1:N
            if n == 1
                % Initial speed constraint: S_x(u, 1) == S_x_init(u)
                c_update(u, n) = S_x(u, n) == S_x_init(u);
            else
                % Speed update constraint: S_x(u, n) == S_x(u, n-1) + phi_x(u, n-1) * delta_t
                c_update(u, n) = S_x(u, n) == S_x(u, n-1) + phi_x(u, n-1) * delta_t;
            end
            % Speed bounds constraints: S_min <= S_x(u, n) <= S_max
            c_lower(u, n) = S_x(u, n) >= S_min;
            c_upper(u, n) = S_x(u, n) <= S_max;
        end
    end
end

% Transmit power constraint definition
function c_upper = transmitPowerConstraint(P, P_max, V, U, K, N)
    % Upper bound constraint
    c_upper = optimconstr(V, U, K, N);
    
    for v = 1:V
        for u = 1:U
            for k = 1:K
                for n = 1:N
                    c_upper(v, u, k, n) = P(v, u, k, n) <= P_max;
                end
            end
        end
    end
end

% Energy constraint definition
% Function definition for energy lower bound constraint
function c = energyLowerConstraint(P, S_x, delta_t, P0, P_i, u_tip, s0, e0, iota, B, U, N)
    % Initialize constraints
    c = optimconstr(U, N);

    % Loop through each UAV and each time slot
    for u = 1:U
        for n = 1:N
            % Calculate the communication energy consumption for UAV u at time slot n
            comm_energy_u_n = sum(sum(P(:, u, :, n) .* delta_t, 3), 1);
            
            % Calculate the flying power
            S = S_x(u, n);
            flying_power_u_n = P0 * (1 + 3 * S^2 / u_tip^2) + P_i * sqrt(1 + S^4 / (4 * s0^4) - S^2 / (2 * s0^2)) + 0.5 * e0 * iota * B * S^3;
            
            % Calculate the flight energy consumption for UAV u at time slot n
            flight_energy_u_n = flying_power_u_n * delta_t;
            
            % Calculate the total energy consumption
            total_energy = comm_energy_u_n + flight_energy_u_n;
            
            % Define lower bound constraint: total energy should be non-negative
            c(u, n) = total_energy >= 0;
        end
    end
end

% Function definition for energy upper bound constraint
function c = energyUpperConstraint(P, S_x, delta_t, P0, P_i, u_tip, s0, e0, iota, B, E_max, U, N)
    % Initialize constraints
    c = optimconstr(U, N);

    % Loop through each UAV and each time slot
    for u = 1:U
        for n = 1:N
            % Calculate the communication energy consumption for UAV u at time slot n
            comm_energy_u_n = sum(sum(P(:, u, :, n) .* delta_t, 3), 1);
            
            % Calculate the flying power
            S = S_x(u, n);
            flying_power_u_n = P0 * (1 + 3 * S^2 / u_tip^2) + P_i * sqrt(1 + S^4 / (4 * s0^4) - S^2 / (2 * s0^2)) + 0.5 * e0 * iota * B * S^3;
            
            % Calculate the flight energy consumption for UAV u at time slot n
            flight_energy_u_n = flying_power_u_n * delta_t;
            
            % Calculate the total energy consumption
            total_energy = comm_energy_u_n + flight_energy_u_n;
            
            % Define upper bound constraint: total energy should not exceed E_max
            c(u, n) = total_energy <= E_max;
        end
    end
end

% Vehicle model wrap function definition
function [xvwrap] = vehicle_model_wrap(V, N, delta_t, mu, sigma_v, lanes, length_of_highway)
    % velocity for each lane
    vel = zeros(lanes, 1);
    for lane = 1:lanes
        vel(lane) = normrnd(mu, sigma_v);
    end
    
    % Initialize positions of vehicles
    xvwrap = zeros(lanes, V, N);
    xvwrap(:, :, 1) = rand(lanes, V) * length_of_highway;

    % Update positions for each time step
    for n = 1:N-1
        for lane = 1:lanes
            xvwrap(lane, :, n+1) = mod(xvwrap(lane, :, n) + vel(lane) * delta_t, length_of_highway);
        end
    end
end
