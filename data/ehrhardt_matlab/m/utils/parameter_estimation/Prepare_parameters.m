function [output, err_matrix] = Prepare_parameters(varargin)
    if iscell(varargin{1}); 
        input = varargin{1}; 
    else
        input = varargin;
    end;
    
    s_in = 'input{1}';
    s_out = 'x1';
    s_reshape = 'x1(:)';
    s_error = 'length(input{1})';
    
    for i = 2 : length(input)
        s_in = [s_in sprintf(', input{%i}', i)];
        s_out = [s_out sprintf(', x%i', i)];
        s_reshape = [s_reshape sprintf(', x%i(:)', i)];
        s_error = [s_error sprintf(', length(input{%i})', i)];
    end

    eval(sprintf('[%s] = ndgrid(%s);',s_out, s_in));
    eval(sprintf('output = [%s];',s_reshape));
    
    if length(input) == 1
        eval(sprintf('err_matrix = cell(%s, 1);',s_error));
    else
        eval(sprintf('err_matrix = cell(%s);',s_error));
    end
end