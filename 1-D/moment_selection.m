function [list2, list3] = moment_selection(L, biased_string)
% Returns lists of useful, distinct moments to compute in 1D.
%
% [list2, list3] = moment_selection(L, biased_string)
%
% L is the length of the signal. biased_string can be either
% 'include biased' or 'exclude biased'. It has to be specified.

    % All useful and distinct moments of order 2
    list2 = (0 : (L-1))';

    % All useful and distinct moments of order 3
    list3 = zeros(L*(L+1)/2, 2);
    k = 0;
    for l1 = 0 : L-1
        for l2 = 0 : l1
            k = k+1;
            list3(k, :) = [l1, l2];
        end
    end

    switch lower(biased_string)
        
        case 'include biased'
            % Nothing to do
            
        case 'exclude biased'
            % Remove biased terms from the lists
            
            biased2 = (list2 == 0);
            list2(biased2) = [];

            biased3 = list3(:, 1) == 0 | ...
                      list3(:, 2) == 0 | ...
                      list3(:, 1) == list3(:, 2);
            list3(biased3, :) = [];
            
        otherwise
            error('Specify the flag: include biased or exclude biased');
            
    end

end
