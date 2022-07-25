function s = rename_fields(s)
    if iscell(s)
        s = cellfun(@(x)rename_fields(x),s,'UniformOutput',false);
        return
    elseif ~isstruct(s)
        return
    end
    f = fieldnames(s);
    for jj = 1:numel(f)
        curr_fieldname = f{jj};
        if curr_fieldname(end) == '2'
            new_fieldname = curr_fieldname(1:end-1);
            [s.(new_fieldname)] = deal(s.(curr_fieldname));
            s = rmfield(s, curr_fieldname);
        end

    end
end