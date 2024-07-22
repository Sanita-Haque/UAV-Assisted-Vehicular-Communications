% Define parameters for simulation
U = 2;  % Number of UAVs
V = 10;  % Number of vehicles
K = 10;  % Number of subchannels
N = 20;  % Number of time slots
xi = 0.3;  % Weighting factor for throughput and energy efficiency
P_max = 11;  % Maximum transmit power
E_max = 1.5e5;  % Maximum energy required for communication
S_min = 25;  % Minimum speed
S_max = 65;  % Maximum speed
delta_t = 0.5;  % Time step
d_min = 10;  % Minimum distance to avoid collision
H = 100;  % UAV height
W = 1e6;  % Bandwidth
sigma = 1e-14;  % Noise power
h0 = 1e-4;  % Channel gain at reference distance
length_of_highway = 1000;
lanes = 1;
mu = 65;
sigma_v = 5;
lambda = 20;


FontSize = 20;

% Parameters for UAV flying power consumption
P0 = 1597.12;
P_i = 88.63;
u_tip = 120;
s0 = 4.03;
e0 = 0.6;
iota = 1.225;
B = 0.503;
max_distance = 400;

% Initial conditions
q_x_init = linspace(0, 500, U);
q_x_init = q_x_init(1:U);
q_y_init = H * ones(1, U); % Constant altitude of 100m for all UAVs

% Incremental Speeds Initialization
increment = (S_max - S_min) / (U + 1);
S_x_init = S_min + (1:U) * increment; % Incremental speeds

% Generate initial vehicle positions based on Poisson distribution
x_v_init = vehicle_model_wrap(V, N, delta_t, mu, sigma_v, lanes, length_of_highway, lambda);

y_v_init = zeros(lanes, V, N); % y-positions of vehicles

% Decision variables
P = optimvar('P', V, U, K, N);
q_x = optimvar('q_x', U, N);
%q_x = optimvar('q_x', U, N, 'LowerBound', 0, 'UpperBound', length_of_highway); % q_x bounded to 0 to 1000 meters
q_y = optimvar('q_y', U, N, 'LowerBound', 100, 'UpperBound', 100); % Constant altitude
S_x = optimvar('S_x', U, N, 'LowerBound', S_min, 'UpperBound', S_max);
phi_x = optimvar('phi_x', U, N-1, 'LowerBound', -300, 'UpperBound', 10);
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
prob.Constraints.userAssocCons2 = userAssociationConstraint2(omega, V, N, U);
%prob.Constraints.userAssocCons = userAssociationConstraint(omega, q_x, x_v_init, H, V, U, N);
prob.Constraints.userAssocCons = userAssociationConstraint(omega, q_x, x_v_init, H, V, U, N, max_distance);
prob.Constraints.subchannelAssignCons = subchannelAssignmentConstraint(alpha, omega, V, U, K, N);
prob.Constraints.antiCollisionCons = antiCollisionConstraint(q_x, q_y, d_min, U, N);
prob.Constraints.positionUpdateCons = dynamicPositionUpdateVelocityMatching(q_x, q_x_init, S_x, delta_t, U, N, [], [], [], phi_x);
%prob.Constraints.positionUpdateCons = dynamicPositionUpdateConstraint(q_x, q_x_init, S_x, phi_x, delta_t, U, N, x_v_init, H, V, d_min);
%prob.Constraints.positionUpdateCons = dynamicPositionUpdateVelocityMatching(q_x, q_x_init, S_x, delta_t, U, N, x_v_init, H, V, d_min,phi_x);
[c_update, c_lower, c_upper] = speedUpdateConstraint(S_x, S_x_init, phi_x, delta_t, U, N, S_min, S_max);
prob.Constraints.speedUpdateCons = c_update;
prob.Constraints.speedLowerCons = c_lower;
prob.Constraints.speedUpperCons = c_upper;
%prob.Constraints.distanceCons = distanceConstraint(q_x, x_v_init, H, U, V, N, max_distance);
prob.Constraints.energyLowerCons = energyLowerConstraint(P, S_x, delta_t, P0, P_i, u_tip, s0, e0, iota, B, U, N);
prob.Constraints.energyUpperCons = energyUpperConstraint(P, S_x, delta_t, P0, P_i, u_tip, s0, e0, iota, B, E_max, U, N);
%prob.Constraints.transmitPowerUpperCons = transmitPowerConstraint(P, P_max, V, U, K, N);
prob.Constraints.couplingCons = couplingConstraint(omega, alpha, V, U, K, N);
prob.Constraints.ensureSubchannelAssignment = ensureSubchannelAssignment(alpha, V, K, N);
prob.Constraints.powerAssocCons = powerAssociationConstraint(P, omega, V, U, K, N, P_max);
prob.Constraints.powerNonNegativeCons = powerNonNegativeConstraint(P, V, U, K, N);

% Initial guess
x0.q_x = repmat(q_x_init', 1, N);
x0.q_y = repmat(q_y_init', 1, N);
x0.S_x = repmat(S_x_init', 1, N);
x0.phi_x = 0.1 * rand(U, N-1) - 0.05;

x0.P = zeros(V, U, K, N); % Initialize with zeros

x0.omega = zeros(V, U, N);
for n = 1:N
    for v = 1:V
        assigned_uav = mod(v, U) + 1;
        x0.omega(v, assigned_uav, n) = 1;
        for k = 1:K
            if x0.omega(v, assigned_uav, n) == 1
                x0.P(v, assigned_uav, k, n) = P_max * rand; % Assign non-zero power to connected pairs
            end
        end
    end
end

% Calculate distances and assign vehicles to the closest UAV
distances = zeros(V, U, N);
for n = 1:N
    for v = 1:V
        for u = 1:U
            distances(v, u, n) = sqrt((q_x_init(u) - x_v_init(1, v, n))^2 + H^2);
        end
    end
end

% Subchannel allocation (alpha): Evenly distribute initial subchannel allocations with minimal usage
x0.alpha = zeros(V, K, N);
for k = 1:K
    for n = 1:N
        assigned_vehicle = mod(k, V) + 1;
        x0.alpha(assigned_vehicle, k, n) = 1; % initial allocation
    end
end

% Pre-optimization calculations for total data rate and energy consumption
total_data_rate_before = zeros(1, N);
total_energy_before = zeros(1, N);

for n = 1:N
    for u = 1:U
        for v = 1:V
            for k = 1:K
                d_uv = sqrt((q_x_init(u) - x_v_init(1, v, n))^2 + (q_y_init(u) - 0)^2 + H^2);
                h_uv = h0 / (d_uv^2);
                snr_uv = h_uv / sigma;
                data_rate_uvkn = 0.9 * W * log2(1 + snr_uv * x0.P(v, u, k, n)); % Data rate before optimization
                total_data_rate_before(n) = total_data_rate_before(n) + data_rate_uvkn * x0.omega(v, u, n) * x0.alpha(v, k, n);
            end
        end
        S = x0.S_x(u, n);
        flying_power_u_n = P0 * (1 + 3 * S^2 / u_tip^2) + P_i * sqrt(1 + S^4 / (4 * s0^4) - S^2 / (2 * s0^2)) + 0.5 * e0 * iota * B * S^3;
        total_energy_before(n) = total_energy_before(n) + sum(sum(x0.P(:, u, :, n), 3), 1) * delta_t + flying_power_u_n * delta_t;
    end
end

options = optimoptions('ga', ...
    'Display', 'iter', ...
    'MaxGenerations', 5, ... % Increase the number of generations
    'FunctionTolerance', 1e-6, ... % Tighten function tolerance
    'ConstraintTolerance', 1e-6, ... % Tighten constraint tolerance
    'PopulationSize', 50, ... % Increase population size for better exploration
    'CrossoverFraction', 0.8, ...
    'MutationFcn', @mutationuniform, ...
    'UseParallel', true); % Use pattern search for hybrid function

% Start parallel pool if not already started
if isempty(gcp('nocreate'))
    parpool;
end
[sol, fval] = solve(prob, x0, 'Solver', 'ga', 'Options', options);

% Define a specific set of colors manually
colors = [0 0.5 0; 0 0 1];


% Plot UAV Speeds Over Time
figure;
hold on;
for u = 1:U
    plot(1:N, sol.S_x(u, :), '-o', 'DisplayName', ['UAV ' num2str(u)], 'Color', colors(u, :), 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', colors(u, :));
end

% Set axis labels, title, and grid
xlabel('Time Slot', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Speed (m/s)', 'FontSize', 20, 'FontWeight', 'bold');
title('UAV Speeds Over Time', 'FontSize', 24, 'FontWeight', 'bold');
set(gca, 'FontSize', 16, 'LineWidth', 1.5);
legend('show', 'Location', 'northwest');
grid on;

% Set the y-axis limit to start from a specific value, e.g., 20
ylim([20, max(sol.S_x, [], 'all')]);

hold off;
% Calculate total data rate and energy consumption after optimization
total_data_rate_after = zeros(1, N);
total_energy_after = zeros(1, N);
data_rate_after = zeros(U, N, V, K); % Store data rate for each u, n, v, k after optimization

for n = 1:N
    for u = 1:U
        for v = 1:V
            for k = 1:K
                d_uv = sqrt((sol.q_x(u, n) - x_v_init(1, v, n))^2 + (sol.q_y(u, n) - 0)^2 + H^2);
                h_uv = h0 / (d_uv^2);
                snr_uv = h_uv / sigma;
                data_rate_uvkn = W * log2(1 + snr_uv * sol.P(v, u, k, n)); % Data rate after optimization
                data_rate_after(u, n, v, k) = data_rate_uvkn; % Store data rate
                total_data_rate_after(n) = total_data_rate_after(n) + data_rate_uvkn * sol.omega(v, u, n) * sol.alpha(v, k, n);
                U1(n) = abs(total_data_rate_after(n)) / 1e9;
            end
        end
        S = sol.S_x(u, n);
        flying_power_u_n = P0 * (1 + 3 * S^2 / u_tip^2) + P_i * sqrt(1 + S^4 / (4 * s0^4) - S^2 / (2 * s0^2)) + 0.5 * e0 * iota * B * S^3;
        total_energy_after(n) = total_energy_after(n) + sum(sum(sol.P(:, u, :, n), 3), 1) * delta_t + flying_power_u_n * delta_t;
        U2(n) = total_energy_after(n) / 1e4;
    end
end

% 2 Plot total data rate before and after optimization
figure;
plot(1:N, total_data_rate_before / 1e6, '-o', 'DisplayName', 'Data Rate Pre Optimization', 'LineWidth', 2);
hold on;
plot(1:N, total_data_rate_after / 1e6, '-x', 'DisplayName', 'Data Rate Post Optimization', 'LineWidth', 2);
xlabel('Time Slot', 'FontSize', 20);
ylabel('Total Data Rate (Mbps)', 'FontSize', 20);
legend('show', 'FontSize', 16);
title('Total Data Rate Before and After Optimization', 'FontSize', 20);
yticks = get(gca, 'YTick');
set(gca, 'YTickLabel', yticks, 'FontSize', 16);
hold off;

% 3 Plot energy consumption before and after optimization
figure;
plot(1:N, total_energy_before / 1e3, '-o', 'DisplayName', 'Energy Consumption Pre Optimization', 'LineWidth', 2);
hold on;
plot(1:N, total_energy_after / 1e3, '-x', 'DisplayName', 'Energy Consumption Post Optimization', 'LineWidth', 2);
xlabel('Time Slot', 'FontSize', 20);
ylabel('Total Energy Consumption (kJ)', 'FontSize', 20);
legend('show', 'FontSize', 16);
title('Total Energy Consumption Before and After Optimization', 'FontSize', 20);
set(gca, 'FontSize', 16);
hold off;

% % 4 Plot UAV Positions Over Time
% figure;
% hold on;
% for u = 1:U
%     plot(sol.q_x(u, :), sol.q_y(u, :), '-o', 'DisplayName', ['UAV ' num2str(u)], 'Color', colors(u, :), 'LineWidth', 2);
% end
% xlim([0, 1000]);
% ylim([0, 150]);
% xlabel('x-position (meters)', 'FontSize', 20);
% ylabel('y-position (meters)', 'FontSize', 20);
% title('UAV Positions Over Time', 'FontSize', 20);
% legend('show', 'FontSize', 16);
% grid on;
% set(gca, 'FontSize', 16);
% hold off;

%Optimized UAV position in each time slot
% for n = 1:N
%     fprintf('Time Slot %d:\n', n);
%     for u = 1:U
%         fprintf('UAV %d: x = %.2f, y = %.2f\n', u, sol.q_x(u, n), sol.q_y(u, n));
%     end
% end

% 5 Plotting of Total Vehicle Associations per UAV Over Time
figure;
uav_associations = squeeze(sum(sol.omega, 1));

% Create a grouped bar chart
b = bar(uav_associations', 'grouped');

% Assign colors to each UAV
for k = 1:length(b)
    b(k).FaceColor = colors(mod(k-1, size(colors, 1)) + 1, :);
end

xlabel('Time Slot', 'FontSize', 20);
ylabel('Total Associations (count)', 'FontSize', 20);
title('Total Vehicle Associations per UAV Over Time', 'FontSize', 20);
legend(arrayfun(@(x) ['UAV ' num2str(x)], 1:U, 'UniformOutput', false), 'Location', 'northoutside', 'FontSize', 16);
set(gca, 'FontSize', 16);
grid on;

% Define distinct colors for each vehicle
colors_vehicle = lines(V); % lines function generates a distinct color for each line, and in this case, each vehicle

% 7 Visualize Subchannel Allocations (alpha) with Bar Graph
figure;
alpha_last_slot = sol.alpha(:, :, end); % Subchannel assignments for the last time slot
bar_data = zeros(V, K);

for v = 1:V
    for k = 1:K
        if alpha_last_slot(v, k) == 1
            bar_data(v, k) = k;
        end
    end
end

% Create a bar graph for each vehicle
hold on;
for k = 1:K
    bar(1:V, bar_data(:, k), 'stacked', 'FaceColor', colors_vehicle(k, :), 'DisplayName', ['Subchannel ' num2str(k)]);
end
hold off;

xlabel('Vehicle', 'FontSize', 20);
ylabel('Assigned Subchannel', 'FontSize', 20);
title('Subchannel Assignments for Each Vehicle in the Last Time Slot', 'FontSize', 20);
legend('show', 'FontSize', 12);
grid on;
set(gca, 'FontSize', 16);
grid on;

% 8 UAV and Vehicle positioning over time
figure;
hold on;
colors_uav = [0 0.5 0; 0 0 1];

% Determine the overall limits based on the entire dataset
x_min = 0;
x_max = 1000;
y_min = -10;
y_max = 200;

% Initialize video writer
myVideo = VideoWriter('UAV_Vehicle_Positions_first_graph.avi');
myVideo.FrameRate = 2;
open(myVideo);

% Loop over each time step to update the plot
for n = 1:N
    % Clear previous scatter plots
    clf;
    hold on;
    
    % Add rectangles for each lane
    for lane = 1:lanes
        rectangle('Position', [2, -2, length_of_highway, 2], 'FaceColor', [0.8, 0.8, 0.8], 'EdgeColor', 'none');
    end
    
    % Plot vehicle positions for the current time step
    for v = 1:V
        scatter(squeeze(x_v_init(1, v, n)), 0, 60, 'filled', 'MarkerFaceColor', colors_vehicle(v, :));
    end

    % Plot UAV positions for the current time step
    for u = 1:U
        scatter(sol.q_x(u, n), sol.q_y(u, n), 50, 'filled', 'MarkerFaceColor', colors_uav(u, :));
        text(sol.q_x(u, n), sol.q_y(u, n), ['UAV' num2str(u)], 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', 'Color', colors_uav(u, :));
    end
    
    % Adjust plot settings
    xlim([x_min, x_max]);
    ylim([y_min, y_max]);
    xlabel('x-position (meters)', 'FontSize', 20);
    ylabel('y-position (meters)', 'FontSize', 20);
    title(sprintf('UAV and Vehicle Positions at Time Step %d', n), 'FontSize', 20);
    grid on;
    set(gca, 'FontSize', 16);
    drawnow;
    
    % Capture the plot as a frame
    frame = getframe(gcf);
    writeVideo(myVideo, frame);
end

% Close the video writer
close(myVideo);
hold off;

for n = 1:N
    for v = 1:V
        % Distance from vehicle v to UAV1
        dist_to_uav1(v, n) = norm([sol.q_x(1, n) - x_v_init(1, v, n), sol.q_y(1, n)]);
        
        % Distance from vehicle v to UAV2
        dist_to_uav2(v, n) = norm([sol.q_x(2, n) - x_v_init(1, v, n), sol.q_y(2, n)]);
    end
end

% Number of time slots per figure
time_slots_per_fig = 10;
num_figs = ceil(N / time_slots_per_fig);

for fig_idx = 1:num_figs
    % Calculate the range of time slots for this figure
    start_slot = (fig_idx - 1) * time_slots_per_fig + 1;
    end_slot = min(fig_idx * time_slots_per_fig, N);
    
    % Extract the relevant data for this range of time slots
    dist_to_uav1_fig = dist_to_uav1(:, start_slot:end_slot);
    dist_to_uav2_fig = dist_to_uav2(:, start_slot:end_slot);
    
    % Create a new figure
    figure;
    
    % Bar graph for distances to UAV1
    subplot(2, 1, 1);
    bar(dist_to_uav1_fig', 'grouped');
    xlabel('Time Slot', 'FontSize', 20);
    ylabel('Distance to UAV1 (m)', 'FontSize', 20);
    title(sprintf('Distance of Vehicles to UAV1 Over Time (Time Slots %d-%d)', start_slot, end_slot), 'FontSize', 20);
    legend(arrayfun(@(x) ['Vehicle ' num2str(x)], 1:V, 'UniformOutput', false), 'Location', 'bestoutside', 'FontSize', 10);
    grid on;
    set(gca, 'FontSize', 16);
    
    % Bar graph for distances to UAV2
    subplot(2, 1, 2);
    bar(dist_to_uav2_fig', 'grouped');
    xlabel('Time Slot', 'FontSize', 20);
    ylabel('Distance to UAV2 (m)', 'FontSize', 20);
    title(sprintf('Distance of Vehicles to UAV2 Over Time (Time Slots %d-%d)', start_slot, end_slot), 'FontSize', 20);
    legend(arrayfun(@(x) ['Vehicle ' num2str(x)], 1:V, 'UniformOutput', false), 'Location', 'bestoutside', 'FontSize', 10);
    grid on;
    set(gca, 'FontSize', 16);
end

% After the optimization process
% Calculate data rate for each vehicle in each time slot
% Initialize the data rate matrix
vehicle_data_rate = zeros(V, N);

% Calculate the data rate for each vehicle over all time slots
for n = 1:N
    for v = 1:V
        total_rate = 0;
        for u = 1:U
            for k = 1:K
                d_uv = sqrt((sol.q_x(u, n) - x_v_init(1, v, n))^2 + H^2);
                h_uv = h0 / (d_uv^2);
                snr_uv = h_uv / sigma;
                data_rate_uvkn = W * log2(1 + snr_uv * sol.P(v, u, k, n));
                total_rate = total_rate + abs(data_rate_uvkn) * sol.omega(v, u, n);
            end
        end
        vehicle_data_rate(v, n) = total_rate / 1e6; % Convert to Mbps
    end
end

% Define the number of time slots per chart
time_slots_per_fig = 10;
num_figs = ceil(N / time_slots_per_fig);

% Plot the data rates as grouped bar charts for each segment of 10 time slots
for fig_idx = 1:num_figs
    % Calculate the range of time slots for this figure
    start_slot = (fig_idx - 1) * time_slots_per_fig + 1;
    end_slot = min(fig_idx * time_slots_per_fig, N);
    
    % Extract the relevant data for this range of time slots
    vehicle_data_rate_segment = vehicle_data_rate(:, start_slot:end_slot);
    
    % Create a new figure
    figure;
    b3 = bar(start_slot:end_slot, vehicle_data_rate_segment', 'grouped');
    xlabel('Time Slot', 'FontSize', 20);
    ylabel('Data Rate (Mbps)', 'FontSize', 20);
    title(sprintf('Data Rate for Each Vehicle Over Time (Time Slots %d-%d)', start_slot, end_slot), 'FontSize', 20);
    
    % Apply colors to bars for data rates
    for k = 1:length(b3)
        b3(k).FaceColor = colors_vehicle(k, :);
    end
    
    % Adjust the legend to be smaller
    legend(arrayfun(@(x) ['Vehicle ' num2str(x)], 1:V, 'UniformOutput', false), ...
        'Location', 'northoutside', 'FontSize', 12, 'NumColumns', ceil(V / 4)); % Adjust the number of columns as needed
    
    set(gca, 'FontSize', 16);
    grid on;
    
    % Optional: Adjust the axis limits if needed
    ylim([0, max(vehicle_data_rate_segment, [], 'all')]);
end


% Objective function
function f_value = ObjectiveFunction(q_x, q_y, P, omega, alpha, S_x, phi_x, x_v_init, y_v_init, xi, W, sigma, h0, U, V, K, N, H, P0, P_i, u_tip, s0, e0, iota, B, delta_t)
    % Scaling factors
    data_rate_scale = 1e9; % Scale for data rate to bring it to a comparable range with energy
    energy_scale = 1e4; % Scale for energy to bring it to a comparable range with data rate

    % Initialize arrays to store U1, U2, and f values for each time step
    U1_values = zeros(1, N);
    U2_values = zeros(1, N);
    f_values = zeros(1, N);

    for n = 1:N
        total_data_rate = 0;
        total_energy = 0;
        regularization = 0;
        
        % Data rate calculation for current time step
        for u = 1:U
            data_rate_u_n = 0;
            energy_u_n = 0;
            for v = 1:V
                for k = 1:K
                    % Distance calculation (dimension handling)
                    d_uv = sqrt((q_x(u, n) - x_v_init(1, v, n))^2 + (q_y(u, n) - y_v_init(1, v, n))^2 + H^2);
                    % Channel power gain
                    h_uv = h0 / (d_uv^2);
                    % Signal-to-noise ratio (SNR)
                    snr_uv = h_uv / sigma;
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

        % Objective function calculation for current time step
        U1 = abs(total_data_rate) / data_rate_scale;
        U2 = total_energy / energy_scale;

        % Store U1 and U2 values for current time step
        U1_values(n) = U1;
        U2_values(n) = U2;
    end

    % Normalize U1 and U2 using min-max normalization
    U1_min = min(U1_values);
    U1_max = max(U1_values);
    U2_min = min(U2_values);
    U2_max = max(U2_values);

    U1_values = (U1_values - U1_min) / (U1_max - U1_min);
    U2_values = (U2_values - U2_min) / (U2_max - U2_min);

    % Calculate the objective function values for each time step using normalized U1 and U2
    for n = 1:N
        f_values(n) = xi * U1_values(n) - (1 - xi) * U2_values(n) - 0.01 * regularization;
    end

    % Display U1 and U2 values for each time step
    
    % Aggregate the objective function values to return a single scalar
    f_value = sum(f_values); % Sum of objective function values over all time steps
    
    % Make sure to maximize the objective function by returning the positive values
    f_value = -f_value; % To maximize, we negate the objective function
end

% User association constraint definition
function c = userAssociationConstraint2(omega, V, N, U)
    c = optimconstr(V, N);
    for v = 1:V
        for n = 1:N
            c(v, n) = sum(omega(v, 1:U, n)) == 1;
        end
    end
end

function c = userAssociationConstraint(omega, q_x, x_v_init, H, V, U, N, max_distance)
    % Create the constraint matrix for user association
    userAssocCons = optimconstr(V, N, U);
    
    % Create the balancing constraint matrix
    %balanceConstraint = optimconstr(U, N);

    % Create the max distance constraint matrix
    maxDistCons = optimconstr(V, U, N);
    
    % Iterate over each vehicle and time slot
    for v = 1:V
        for n = 1:N
            % Initialize an array to store distances from each UAV
            dist = optimexpr(U, 1);
            
            % Calculate the distance from each UAV to the vehicle
            for u = 1:U
                dist(u) = sqrt((q_x(u, n) - x_v_init(1, v, n))^2 + H^2);
            end
            
            % Ensure that only one UAV is associated with each vehicle and
            % the distance is within the max allowable distance
            for u = 1:U
                userAssocCons(v, n, u) = omega(v, u, n) * dist(u) <= sum(omega(v, :, n) .* dist');
                maxDistCons(v, u, n) = omega(v, u, n) * dist(u) <= omega(v, u, n) * max_distance;
            end
        end
    end
    
%     % Add balancing constraints
%     for u = 1:U
%         for n = 1:N
%             % Balance the number of vehicles assigned to each UAV
%             balanceConstraint(u, n) = sum(omega(:, u, n)) <= (V / U) + 1;
%         end
%     end
    
    % Combine the constraints
    %c = [userAssocCons(:); balanceConstraint(:); maxDistCons(:)];
    c = [userAssocCons(:); maxDistCons(:)];
end


function c = subchannelAssignmentConstraint(alpha, ~, ~, ~, K, N)
    c = optimconstr(K, N);
    for k = 1:K
        for n = 1:N
            % Ensure each subchannel is assigned to at most one vehicle per time slot
            c(k, n) = sum(alpha(:, k, n)) <= 1;
        end
    end
end

function c = ensureSubchannelAssignment(alpha, V, K, N)
    c = optimconstr(V, N);
    for v = 1:V
        for n = 1:N
            c(v, n) = sum(alpha(v, :, n)) >= 1;
        end
    end
end

function c = couplingConstraint(omega, alpha, V, U, K, N)
    c = optimconstr(K, U, N);
    for k = 1:K
        for u = 1:U
            for n = 1:N
                c(k, u, n) = sum(omega(1:V, u, n) .* alpha(1:V, k, n)) <= 1;
            end
        end
    end
end

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
function c = dynamicPositionUpdateVelocityMatching(q_x, q_x_init, S_x, delta_t, U, N, ~, ~, ~, phi_x)
    c = optimconstr(U, N);
    for u = 1:U
        for n = 1:N
            if n == 1
                c(u, n) = q_x(u, n) == q_x_init(u);  % Set initial position
            else
                % Update UAV position based on the provided equation
                c(u, n) = q_x(u, n) == q_x(u, n-1) + S_x(u, n-1) * delta_t + 0.5 * phi_x(u, n-1) * delta_t^2;
            end
        end
    end
end
% function c = dynamicPositionUpdateVelocityMatching(q_x, q_x_init, S_x, delta_t, U, N, x_v_init, ~, ~, ~,phi_x)
%     c = optimconstr(U, N);
%     for u = 1:U
%         for n = 1:N
%             if n == 1
%                 c(u, n) = q_x(u, n) == q_x_init(u);  % Set initial position
%             elseif n == 2
%                 % Only have one previous time step for n=2
%                 avg_velocity = (mean(x_v_init(1, :, n-1)) - mean(x_v_init(1, :, 1))) / delta_t;
%                 c(u, n) = q_x(u, n) == q_x(u, n-1) + avg_velocity * delta_t + S_x(u, n-1) * delta_t + 0.5 * (phi_x(u, n - 1) * delta_t^2);
%             else
%                 % Calculate the average velocity of the vehicles
%                 avg_velocity = (mean(x_v_init(1, :, n-1)) - mean(x_v_init(1, :, n-2))) / delta_t;
%                 % Update UAV position based on velocity matching
%                 c(u, n) = q_x(u, n) == q_x(u, n-1) + avg_velocity * delta_t + S_x(u, n-1) * delta_t + 0.5 * (phi_x(u, n - 1) * delta_t^2);
%             end
%         end
%     end
% end

function [c_update, c_lower, c_upper] = speedUpdateConstraint(S_x, S_x_init, phi_x, delta_t, U, N, S_min, S_max)
    c_update = optimconstr(U, N); % Initialize the speed update constraint matrix
    c_lower = optimconstr(U, N);  % Initialize the lower bound constraint matrix
    c_upper = optimconstr(U, N);  % Initialize the upper bound constraint matrix
    
    for u = 1:U
        for n = 1:N
            if n == 1
                c_update(u, n) = S_x(u, n) == S_x_init(u);
            else
                c_update(u, n) = S_x(u, n) == S_x(u, n-1) + phi_x(u, n-1) * delta_t;
            end
            c_lower(u, n) = S_x(u, n) >= S_min;
            c_upper(u, n) = S_x(u, n) <= S_max;
        end
    end
end

function c = powerNonNegativeConstraint(P, V, U, K, N)
    c = optimconstr(V, U, K, N);
    for v = 1:V
        for u = 1:U
            for k = 1:K
                for n = 1:N
                    c(v, u, k, n) = P(v, u, k, n) >= 0;
                end
            end
        end
    end
end

function c = powerAssociationConstraint(P, omega, V, U, K, N, P_max)
    c = optimconstr(V, U, K, N);
    for v = 1:V
        for u = 1:U
            for k = 1:K
                for n = 1:N
                    c(v, u, k, n) = P(v, u, k, n) <= omega(v, u, n) * P_max;
                end
            end
        end
    end
end

function c = energyLowerConstraint(P, S_x, delta_t, P0, P_i, u_tip, s0, e0, iota, B, U, N)
    c = optimconstr(U, N);

    for u = 1:U
        for n = 1:N
            comm_energy_u_n = sum(sum(P(:, u, :, n) .* delta_t, 3), 1);
            
            S = S_x(u, n);
            flying_power_u_n = P0 * (1 + 3 * S^2 / u_tip^2) + P_i * sqrt(1 + S^4 / (4 * s0^4) - S^2 / (2 * s0^2)) + 0.5 * e0 * iota * B * S^3;
            
            flight_energy_u_n = flying_power_u_n * delta_t;
            
            total_energy = comm_energy_u_n + flight_energy_u_n;
            
            c(u, n) = total_energy >= 0;
        end
    end
end

function c = energyUpperConstraint(P, S_x, delta_t, P0, P_i, u_tip, s0, e0, iota, B, E_max, U, N)
    c = optimconstr(U, N);

    for u = 1:U
        for n = 1:N
            comm_energy_u_n = sum(sum(P(:, u, :, n) .* delta_t, 3), 1);
            
            S = S_x(u, n);
            flying_power_u_n = P0 * (1 + 3 * S^2 / u_tip^2) + P_i * sqrt(1 + S^4 / (4 * s0^4) - S^2 / (2 * s0^2)) + 0.5 * e0 * iota * B * S^3;
            
            flight_energy_u_n = flying_power_u_n * delta_t;
            
            total_energy = comm_energy_u_n + flight_energy_u_n;
            
            c(u, n) = total_energy <= E_max;
        end
    end
end

function [xvwrap] = vehicle_model_wrap(V, N, delta_t, mu, sigma_v, lanes, length_of_highway, lambda)
    % Initialize velocities for each lane
    vel = zeros(lanes, 1);
    for lane = 1:lanes
        vel(lane) = normrnd(mu, sigma_v);
    end
    
    % Initialize vehicle positions based on Poisson arrivals
    xvwrap = zeros(lanes, V, N); % Use zeros instead of NaN
    current_vehicle = 1;
    for n = 1:N
        if current_vehicle <= V
            num_arrivals = poissrnd(lambda * delta_t); % Number of arrivals in each time slot
            for a = 1:num_arrivals
                if current_vehicle <= V
                    lane = randi(lanes); % Random lane assignment
                    xvwrap(lane, current_vehicle, n) = rand * length_of_highway; % Random position along the highway
                    current_vehicle = current_vehicle + 1;
                end
            end
        end
    end

    % Ensure all vehicles have initial positions
    for lane = 1:lanes
        for v = 1:V
            if xvwrap(lane, v, 1) == 0
                xvwrap(lane, v, 1) = rand * length_of_highway;
            end
        end
    end

    % Propagate positions over time
    for n = 1:N-1
        for lane = 1:lanes
            for v = 1:V
                xvwrap(lane, v, n+1) = mod(xvwrap(lane, v, n) + vel(lane) * delta_t, length_of_highway);
            end
        end
    end
end
