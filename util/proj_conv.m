function [x] = proj_conv(y, c)
p = sign(y);
% a = (sum(abs(y)) - c) / length(y);
% x = y - a * p;
x = proj_simplex(repmat(abs(y)', 2, 1), c);
x = x(1, :);
x = x' .* p;
end