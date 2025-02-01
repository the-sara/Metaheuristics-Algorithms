function [x_best, f_best] = simulated_annealing(func, x_init, T0, max_iter, lb, ub, alpha)

    % Initialize variables
    x_current = x_init;            
    f_current = func(x_current(1), x_current(2)); 
    x_best = x_current;           
    f_best = f_current;           
    T = T0;                       

    for k = 1:max_iter
        % Generate a neighboring solution
        x_new = x_current + rand(1, length(x_current)) .* (ub - lb);     
        f_new = func(x_new(1), x_new(2));

        % The test
        delta_f = f_new - f_current;

        if delta_f < 0 || rand() < exp(-delta_f / T)
            x_current = x_new;
            f_current = f_new;
            if f_current < f_best  
                x_best = x_current;
                f_best = f_current;
            end
        end
        
        % Update temperature
        T = T0 / (1 + alpha * k);
    end
end
