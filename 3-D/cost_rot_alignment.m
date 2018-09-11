function f = cost_rot_alignment(a_lms, a_lms_ref, Rxyz)
% Cost 
R_l = RN2RL(getSHrotMtx(Rxyz, length(a_lms)-1, 'complex'));
a_lms_rot = cellfun(@mtimes, a_lms, R_l, 'UniformOutput', 0);

f = norm(vec_cell(a_lms_rot) - vec_cell(a_lms_ref))^2/norm(vec_cell(a_lms_ref))^2;