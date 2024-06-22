function [x] = vehicle_model_wrap(num_vehicles,N,dt,mu,sigma,lanes,length_of_highway)

% velocity for each lane
v = zeros(lanes,1);
for lane = 1:lanes
    v(lane) = normrnd(mu(lane),sigma(lane));
end
x = zeros(lanes,num_vehicles,N);
x(:,:,1) = rand(lanes,num_vehicles) * length_of_highway;

% for each time n
for n = 1:N-1

    % for each lane
    for lane = 1:lanes
        x(lane,:,n+1) = mod( x(lane,:,n) + v(lane)*dt, length_of_highway );
    end
end
end