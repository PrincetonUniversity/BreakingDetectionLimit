function [Rots, pts_on_S2] = sample_S2(num_pts)
% Generate rotation matrices corresponding to viewing directions forming an
% approximately uniform mesh on the sphere, and a random in-plane rotation.
% 
% Inputs:
%   * num_pts: number of points on the sphere to form the mesh
% 
% Outputs:
%   * Rots: stack of 3x3 rotation matrices corresponding to viewing
%   directions forming an approximately uniform mesh on the sphere and a
%   random in-plane rotation.
%   * pts_on_s2: matrix of size num_pts x 3 with the actual points on the
%   sphere in R^3.
% 
% Eitan Levin, August 2018

% use manopt to generate points on the sphere forming an approximately
% uniform mesh:
pts_on_S2 = packing_on_the_sphere(3, num_pts); 

Rots = zeros(3, 3, num_pts, 'double');
for ii = 1:num_pts
    % For each point, generate a rotation matrix rotating the z-axis to the
    % given point:
    v1 = pts_on_S2(ii, :);
    v2 = randn(1, 3); 
    v2 = v2/norm(v2);
    v2 = v2 - dot(v1, v2)*v1;
    v2 = v2/norm(v2);
    v3 = cross(v1, v2);
    R = [v3; v2; v1].';
    if det(R) < 1, R(:, [1,2]) = R(:, [2,1]); end
    Rots(:,:,ii) = R;
end
