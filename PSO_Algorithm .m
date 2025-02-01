function [x_best, f_best] = particle_swarm_optimization(func, kmax)

    % Initialize parameters
    w = 1;           % Inertia weight
    c1 = 0.8;            % Cognitive acceleration coefficient
    c2 = 1.4;            % Social acceleration coefficient
    lb = -4;            % Lower bound
    ub = 4;          % Upper bound
    num_particles = 4; % Number of particles
    dim = 1;           % Dimensionality of the problem

    % Initialize particle positions and velocities
    %xis = lb + (ub - lb) .* rand(num_particles, dim);
    xis=[-2,0,1,3]
    vis = zeros(num_particles, dim); % Initial velocities

    % Evaluate the initial function values
    func_values = func(xis); 
    Pbests = xis; % Personal best positions

    % Find the global best
    [vbest, idx] = min(func_values);
    gbest_current = xis(idx);

    % Optimization loop
    for k = 1:kmax
        for i = 1:num_particles
            r1 = 0.3;
            r2 = 0.4;

            % Update velocity and position
            vis(i) = w * vis(i) + c1 * r1 * (Pbests(i) - xis(i)) + c2 * r2 * (gbest_current - xis(i));
            xis(i) = xis(i) + vis(i);

            % Boundary control
            xis(i) = max(min(xis(i), ub), lb);

            % Evaluate the objective function
            func_values(i) = func(xis(i));

            % Update personal best
            if func_values(i) < func(Pbests(i))
                Pbests(i) = xis(i);
            end
        end

        % Update global best
        [testvbest, index] = min(func_values);
        if testvbest < vbest
           vbest = testvbest;
           gbest_current = xis(index);
        end
    end

    % Return the best solution
    x_best = gbest_current;
    f_best = vbest;
end

% Define the objective function
f = @(x) x.^5 - 5 .* x.^3  - 20.*x +5;

% Call the PSO function
[x_best1, f_best1] = particle_swarm_optimization(f, 2);
disp('Best solution is:');
disp(x_best1);
disp('Best function value is:');
disp(f_best1);
