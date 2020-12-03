function [s, s_label] = Similarity(x, x0, roi)

    fn = fieldnames(roi);
    
    s = zeros(length(fn)+1,1);
    s_label = cell(length(fn)+1,1);
    
    s(1) = Rel_l2(x, x0);
    s_label{1} = 'l2whole';
        
    for i = 1 : length(fn)
        eval(sprintf('s(i+1) = Rel_l2(x, x0, roi.%s);', fn{i}));
        s_label{i+1} = ['l2' fn{i}];
    end
        
end