function exitcode = reformat_runner(filelist_path)

    % create config struct, change this for inhib data.
    config = {};
    config.exc_input = true;
    config.pre_stim_len = 99; 
    config.post_stim_len = 800;

    fid = fopen(filelist_path,'rt');
    while true
        
        % load next dataset
        dataset_path = fgetl(fid);
        if ~ischar(dataset_path); break; end  %end of file
        disp('Now processing:');
        disp(dataset_path);

        mat_contents = load(dataset_path);

        % make savepath using filename
        [~, name, ~] = fileparts(dataset_path);
        savepath = strcat(name, '_cmReformat.mat');
        disp('Saving file to:');
        disp(savepath);

        % call cmReformat script
        try
            cmReformat(mat_contents, savepath, config);
        catch
            disp('Error reformating dataset at: ');
            disp(dataset_path);
            disp('Continuing...');
        end


    end
    fclose(fid);
    exitcode = 0;
end