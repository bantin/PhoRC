
function [new_targets, new_rois] = reformat_grid_targets(ExpStruct)
    
    targets = ExpStruct.holoRequest.targets;
    new_targets = unique(targets, 'rows');


    % take old rois from experiment, find  that pnt in new targets list ( artificial), insert it
    % the the new rois list which is used to costruct stimuli matrix;
    new_rois=cell(size(ExpStruct.holoRequest.rois,1),1);
    for i=1: size(ExpStruct.holoRequest.rois,1)

        this_holo_members=ExpStruct.holoRequest.rois{i,1};

        for j=1:size(this_holo_members,2)

            x = targets(this_holo_members(j),1); % find coords of old/experimental holorequest
            y = targets(this_holo_members(j),2);
            z = targets(this_holo_members(j),3);

           % find which point in new holoRequest has that coordinate 

           this_point_new_id =  find(new_targets(:,1)==x & new_targets(:,2)==y & new_targets(:,3)==z);

           new_rois{i,1}(1,j)= this_point_new_id;
        end

    end
    
end % endfunction