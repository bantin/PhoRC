function [pscs, psps, stimulus_matrix, targets, img, pipette_position, num_pulses_per_holo] = cmReformat(mat_contents, savepath, config)
    
if ~isfield(mat_contents, 'ExpStruct') % working with weird structure
    ExpStruct = mat_contents.ExpStruct2;
    ExpStruct = rename_fields(ExpStruct);
else
    ExpStruct = mat_contents.ExpStruct;
end

% config determines whether we're working with exc or inhib connections.
disp('Running with config:')
disp(config)
exc_input = config.exc_input;
post_stim_len = config.post_stim_len;
pre_stim_len = config.pre_stim_len;

% Check whether the target structure is faulty and restructure if so
% Prefer to use the actualTargets field if it's there.
if isfield(ExpStruct.holoRequest, 'actualtargets')
    orig_targets = ExpStruct.holoRequest.actualtargets;
else
    orig_targets = ExpStruct.holoRequest.targets;
end

assert(size(orig_targets, 2) == 3, 'Expect targets to have second dimension of size 3')
alt_roi_structure = false;
unique_targets_len = size(unique(orig_targets, 'rows'), 1);
orig_targets_len = size(orig_targets, 1);

if orig_targets_len ~= unique_targets_len
    alt_roi_structure = true;
    [new_targets, new_rois] = reformat_grid_targets(ExpStruct);
    ExpStruct.holoRequest.targets = new_targets;
end


%% Format data for circuit mapping software pipeline
nPlanes=unique(ExpStruct.holoRequest.targets(:,3));

% for plane=1:size(ExpStruct.holoRequest.stack,2)
%     figure(10000+plane); %figure1-4
%     imagesc(ExpStruct.holoRequest.stack{1, plane})
%     %img=imread(fullfile(['D:\synaptic_mapping_results\targeted_protocols\Chrome2F_PVCre\map_results\06232021_cell5map\plane',num2str(zDepths(plane)),'_Composite.jpg']));
%     %imagesc(img); hold on;
%     hold on;
%     title(['plane image: ',num2str(nPlanes(plane))]);
%     ntargets_per_plane{:,plane}=find(ExpStruct.holoRequest.targets(:,3)==nPlanes(plane,1));
%     for k=(ntargets_per_plane{:,plane})
%         text(ExpStruct.holoRequest.targets(k,2)+5,ExpStruct.holoRequest.targets(k,1),num2str(k),'Color','w');
%         hold on; %number displayed 5 pixels on the right of the cell
%         scatter(ExpStruct.holoRequest.targets(k,2),ExpStruct.holoRequest.targets(k,1),10,'filled','b');
%     end
% 
% end
%%
%Borrow sections from Masato code - organising structure

%% 1. Set up struct for sorted data (SortedData)
tic;
% SortedData.stack=ExpStruct.holoRequest.stack;
SortedData.mouseID = ExpStruct.mouseID;
SortedData.reps=ExpStruct.expParams.repeats;
%SortedData.multiclamp_Rs_log = ExpStruct.multiclamp_Rs_log;
%SortedData.Rm = ExpStruct.Rm;
%SortedData.Rs = ExpStruct.Rs;
SortedData.cell_paramneters_execution_times = ExpStruct.cell_paramneters_execution_times;
%SortedData.OnePpulse_inputs = ExpStruct.OnePpulse_inputs;

if isfield(ExpStruct, 'OnePpulse')
    
    SortedData.OnePpulse = ExpStruct.OnePpulse;
else
    SortedData.OnePulse = ExpStruct.OnePpulse_inputs;
end
    
SortedData.expParams = ExpStruct.expParams;
SortedData.holoRequest = ExpStruct.holoRequest;
SortedData.holoStimParams = ExpStruct.holoStimParams;
SortedData.daqParams = ExpStruct.daqParams;
SortedData.inputs = ExpStruct.inputs;
SortedData.trialCond = ExpStruct.trialCond;
%SortedData.cellcheck_ephys_inputs = ExpStruct.cellcheck_ephys_inputs;
SortedData.outParams = ExpStruct.outParams;
SortedData.reps = ExpStruct.expParams.repeats;
directory = 'D:\Data\Synaptic Mapping\Analysis Results\';
fileName = ['Sorted Data ', num2str(ExpStruct.mouseID), '.mat'];

%% 2. Set up variables

% The basic stuff
trialTime = ExpStruct.daqParams.maxSweepLengthSec;
numTrials = length(ExpStruct.trialCond); % ALTERNATIVELY "length(find(cellfun(@isempty, ExpStruct.inputs)==0))". Instead of "length(ExpStruct.inputs)" this puts out true number of trials successfully recorded
powers = unique(ExpStruct.outParams.power);
nPowers = length(ExpStruct.outParams.power);
nConditions=length(unique(ExpStruct.trialCond));
nHolos = ExpStruct.holoStimParams.nHolos; % number of holograms in grid

% Spatial coordinates of spots in grid
SpotCoordinates = ExpStruct.holoRequest.targets;

% Holo spot sizes (bin sizes)
xSpotSize = unique(diff(unique(SpotCoordinates(:,1)))); % dist between centers of spots in x direction
ySpotSize = unique(diff(unique(SpotCoordinates(:,2)))); % dist between centers of spots in y direction
zSpotSize = unique(diff(unique(SpotCoordinates(:,3)))); % dist between centers of spots in z direction

% Grid dimensions (in # of spots)
xSpotNumbers = length(unique(SpotCoordinates(:,1)));
ySpotNumbers = length(unique(SpotCoordinates(:,2)));
zSpotNumbers = length(unique(SpotCoordinates(:,3)));

% Holograms per plane
hologramsPerPlane = nHolos/zSpotNumbers;

% Plane depths
zDepths = unique(SpotCoordinates(:,3));

% Number of planes
nPlanes = length(zDepths);

% Sampling rate
srate = ExpStruct.daqParams.Fs;

% Filter settings
% Extra SG-filter is applied directly after butter filtering. See below.
lpCut = 2000; % filtering data params
[blp,alp] = butter(4, [lpCut/srate],'low');



% %% 3. Plot 1P data- check if cell was opsin positive or negative
% % 
%  if isfield(ExpStruct,'OnePpulse_inputs_cell1')
%      
%      ExpStruct.OnePpulse_inputs=ExpStruct.OnePpulse_inputs_cell1;
%      ExpStruct.OnePpulse_inputs=[ExpStruct.OnePpulse_inputs;ExpStruct.OnePpulse_inputs_cell2];
%  end
%      
% 
% for i = 1:size(ExpStruct.OnePpulse_inputs, 1);
%     t = (linspace(0,(length(ExpStruct.OnePpulse_inputs)/srate),(length(ExpStruct.OnePpulse_inputs))))*1000;
%     
% figure(11)
% plot(t, filtfilt(blp,alp,ExpStruct.OnePpulse_inputs(i,:))) %11/20/2020 added medfilt1, and /10 for 5gOhm feedback resistor adjustment
% title('Current at 1P stimulation');
% xlabel('time [ms]');
% ylabel('Current after 1P stim [nA]');
% hold on;
% legend('1ms pulse','3ms pulse', '5ms pulse');
% xlim([0 2000])
% 
% % zoomed in version
% figure (12)
% xline(50,'color','b','linestyle','--');
% plot(t, filtfilt(blp,alp,ExpStruct.OnePpulse_inputs(i,:))) %11/20/2020 added medfilt1, and /10 for 5gOhm feedback resistor adjustment
% title('1P first pulse');
% xlabel('time [ms]');
% ylabel('Current after 1P stim [nA]');
% hold on;
% 
% xlim([49 57])
% ylim([-1.5 0.05])
% 
% 
% end


%% 4. Separate input data and trial indices according to power "inputs"

inputs = cell(1,nConditions); % isn't used for the rest of this code, but it's here just in case 


if isfield(ExpStruct,'inputsLP')
    inputsLP = cell(1,nConditions);
end

inputsIndices = cell(1,nConditions); %trial numbers done for each power, later used to de-randomize

 if isfield(ExpStruct,'inputs_cell1')
     
     ExpStruct.inputs = ExpStruct.inputs_cell1;
     ExpStruct.inputsLP = ExpStruct.inputs_cell2;
     ExpStruct.inputsLP = ExpStruct.inputsLP';

 end

 
% [Ben]
% In order for the following lines to work, ExpStruct.trialCond must be a
% row vector, so ensure that's the case (some datasets are formatted
% differently)
if ~isrow(ExpStruct.trialCond)
    ExpStruct.trialCond = ExpStruct.trialCond';
end

% [MARCUS]
% By default crop no trials
if ~exist('expt_crop_time', 'var')
    expt_crop_time = size(ExpStruct.trialCond, 2);
    disp('No variable `expt_crop_time` supplied, so data will not be cropped to eliminate patch failures')
end
 
ExpStruct.inputs = ExpStruct.inputs(1:expt_crop_time); % [MARCUS] crop experiment due to patch failure
if isfield(ExpStruct, 'inputsLP')
    ExpStruct.inputsLP = ExpStruct.inputsLP(1:expt_crop_time); %
end

ExpStruct.trialCond = ExpStruct.trialCond(1, 1:expt_crop_time); %
 
for pp = 1:nConditions 
        
    inputs{1, pp} = ExpStruct.inputs(find(ExpStruct.trialCond==pp)); % rearrange the trials based on powers
    inputsIndices{1, pp} = find(ExpStruct.trialCond==pp);
    if isfield(ExpStruct,'inputsLP')
        inputsLP{1,pp}=ExpStruct.inputsLP(find(ExpStruct.trialCond==pp));
    end

end

%inputs = horzcat(inputs{:});

% %% Plot traces acording to condition
% 
% for pp = 1:nConditions
%    figure(15+pp);
%    set(gcf, 'Position', [300 700 1500 500]);
%    t = linspace(0,(length(ExpStruct.outParams.stimLaserPowerOut{pp, 1}(:,end))/srate),(length(ExpStruct.outParams.stimLaserPowerOut{pp, 1}(:,end))));
%    plot(t, ExpStruct.outParams.stimLaserPowerOut{pp, 1}(:,end)*10,'Color', [0.8 0.8 0.8]);hold on;
%    for trial=1:length(inputsIndices{1, pp})
%       
%         plot(t,filtfilt(blp,alp,cell2mat(inputs{1, pp}(trial))),'DisplayName', num2str(inputsIndices{1, pp}(1,trial))); hold on;
%         legend();
%         if isfield(ExpStruct,'inputsLP')
%         
%         plot(t,filtfilt(blp,alp,cell2mat(inputsLP{1, pp}(trial)))); hold on;
%         end
%        
%    end
%  
%    title(['Raw trials -condition: ', num2str(pp)]) ;
%    xlabel('time (s)')
%    ylabel('trace - potential');
%     
%     
% end


%% 5. Break apart each continuous trace into stim windows and rearrange according to hologram sequence  %this needs to be corrected acording to the conditions
disp('Now sorting traces according to power and true hologram sequence...')

% 5.1 Results in a cell array of nHolos x nPowers, and then an additional matrix of the means of each cell in nHolos x powers
wholeTracesSortedByHolos = cell(1, nConditions); % cell for collecting all traces for each holo across powers
datawinsSortedByHolosAllPowers = cell(nConditions, 1);
wholeTracesSortedByHolos_LP = cell(1, nConditions); % cell for collecting all traces for each holo across powers
datawinsSortedByHolosAllPowers_LP = cell(nConditions, 1);

% Detrending parameters
minmax_window = 7000;
gauss_window = 200;

for pp = 1:nConditions % From a select Condition...
    
    this_seq_nPulses=ExpStruct.outParams.nPulses(1,pp);
    datawinsSortedByHolosPerPower = cell(1, nHolos(pp)*this_seq_nPulses); %cell for datawindows sorted by true hologram order, and rounded up across trials and powers
    datawinsSortedByHolosPerPower_LP = cell(1, nHolos(pp)*this_seq_nPulses); %temp_datawinsSortedByHolosAllPowers = cell(1, nHolos(pp));

    % Set time span of each hologram interval "dataWin"(in ms) of start of
    % pulse up to start of next (ipi): diffrentiate the various ipi for
    % different conditions
    dataWin = 0:(ExpStruct.outParams.ipi(pp)+7);%*ExpStruct.outParams.nPulses); % specify the data window (in ms) of inter pulse interval. If changed then apply changes to plotting too
    draw_dataWin = 0:ExpStruct.outParams.ipi(pp)+7;%*ExpStruct.outParams.nPulses;
    dataWinSamplingPnts = dataWin(1):dataWin(end)*srate/1000; % number of sampling points within the ipi
    draw_dataWinSamplingPnts = 0:post_stim_len;

        % Set time span of whole grid stimulation "holoWin"(in ms)
    holoTriggers = ExpStruct.outParams.nextHoloStims{1 , pp}; % when (in fs) during each trial stim laser is on
    holoWin = find(holoTriggers); % indices of holotriggers when hologram is on
    holoTrigOn = find(diff(ExpStruct.outParams.nextHoloStims{1,pp})==1)+1; %time (in fs) when laser is triggered (total number of points should be same as total number of holograms with repetition)
%         if(isempty(holoWin))
%             holoWin = find(ExpStruct.outParams.nextHoloStims{end});
%             holoTrigOn = find(diff(ExpStruct.outParams.nextHoloStims{end})==1)+1;
%         end
   % holoTrigOn=holoTrigOn(1:(SortedData.outParams.nPulses(pp)):end,:);%Further I plot according to multiplexed IPI - so plot the whole period of time during which single holo was presented for a train stimulation
    holoWin = [holoWin(1),holoWin(end)]; % the start and completion times of whole hologram grid stimulation
    
    holoTraces = cell(1, length(inputsIndices{1, pp})); % cell for collecting traces from all trials broken and sorted according to hologram order
    sortedholoTracesAllTrials = cell(length(inputsIndices{1, pp}), nHolos(pp)*this_seq_nPulses); % cell for all data window traces across all trials sorted according to hologram order     
    
    if isfield(ExpStruct,'inputsLP')
        holoTraces_LP = cell(1, length(inputsIndices{1, pp})); % cell for collecting traces from all trials broken and sorted according to hologram order
        sortedholoTracesAllTrials_LP = cell(length(inputsIndices{1, pp}), nHolos(pp)*this_seq_nPulses); 
    end
    
    for tt = 1:length(inputsIndices{1, pp}) % and the trials done for select power...
        trace = ExpStruct.inputs{inputsIndices{1, pp}(tt)}; % take a whole trace from a trial (for select power)...
%         trace = trace-mean(trace(1:holoWin(1))); %clean it up...
        trace = filtfilt(blp, alp, trace); % and add filter...
        if isfield(ExpStruct,'inputsLP')
            trace_LP=ExpStruct.inputsLP{inputsIndices{1, pp}(tt)};
            trace_LP = filtfilt(blp, alp, trace_LP); 
        end
        
        % Add detrending step
        if exc_input
            trace = -trace;
        end
        trace_smooth = smoothdata(trace, 'gaussian', gauss_window);
        trace_min = movmin(trace_smooth, minmax_window, 1);
        baseline = movmax(trace_min, minmax_window, 1);
        trace = trace - baseline;
    
        if isfield(ExpStruct,'inputsLP')
            trace_smooth = smoothdata(trace_LP, 'gaussian', gauss_window);
            trace_min = movmin(trace_smooth, minmax_window, 1);
            baseline = movmax(trace_min, minmax_window, 1);
            trace_LP = trace_LP - baseline;
        end
%         trace_baseline(tt,pp) = mean(trace(1:1000,:));
%         trace_baselineSTD(tt,pp) = std(trace(1:1000,:));
        % call up hologram sequence for select trial...
        holoSequence = (ExpStruct.outParams.sequenceThisTrial{1, inputsIndices{1, pp}(tt)});
        %this_seq_nPulses=ExpStruct.outParams.nPulses(1,pp);
        %disp(this_seq_nPulses);
%         if(isempty(holoSequence)) % if data contains 0mW trials
%             holoSequence = ExpStruct.outParams.sequence{end}(randperm(length(ExpStruct.outParams.sequence{end}))); % generates a random holoSequence for analyzing 0mW trials, useful if we want to use 0mW for a baseline measurment per hologram        
%         end
        
        % break apart a whole trial into data window including pulse and ipi and then sort according to the hologram sequence specific to that trial    
        tempHoloTrace = cell(nHolos(pp)*this_seq_nPulses, 1); % trace broken down into ipis
        sortedHoloTrace = cell(nHolos(pp)*this_seq_nPulses, 1); % 
        if isfield(ExpStruct,'inputsLP')
            tempHoloTrace_LP = cell(nHolos(pp)*this_seq_nPulses, 1); % trace broken down into ipis
            sortedHoloTrace_LP = cell(nHolos(pp)*this_seq_nPulses, 1);
        end
        
        for hh = 1:nHolos(pp)*this_seq_nPulses
            %tracedataWin = trace(holoTrigOn(hh):(holoTrigOn(hh)+ (draw_dataWinSamplingPnts(end)*this_seq_nPulses)));
            tracedataWin = trace(holoTrigOn(hh) - pre_stim_len:(holoTrigOn(hh)+ (draw_dataWinSamplingPnts(end))));
            if isfield(ExpStruct,'inputsLP')
                tracedataWin_LP = trace_LP(holoTrigOn(hh) - pre_stim_len:(holoTrigOn(hh)+ (draw_dataWinSamplingPnts(end))));
            end
            % trace broken down into data windows
            %disp(length(tracedataWin));
            
           
%             tracedataWin = tracedataWin-(mean(tracedataWin(1:100))); % further normalizes based on first 3ms of data window
%             tracedataWin_LP = tracedataWin_LP-(mean(tracedataWin_LP(1:100))); 
            %tracedataWin = sgolayfilt(tracedataWin-mean(tracedataWin(1:60)), 3, 81); % Altnernate to above with added SG-filter
            tempHoloTrace{hh, 1} = tracedataWin; %put the data windows back together this time into one array
            [~,holoSort]=sort(holoSequence'); % get the order of hologram sequence for the trial
            sortedHoloTrace = tempHoloTrace(holoSort)'; % sort the data windows in array according to the trial's hologram sequence
            if isfield(ExpStruct,'inputsLP')
                tempHoloTrace_LP{hh,1}=tracedataWin_LP;
                sortedHoloTrace_LP = tempHoloTrace_LP(holoSort)'; 
            end
        end
        
        % collect all the trials and their traces sorted according to hologram sequence
        sortedholoTracesAllTrials{tt} = vertcat(sortedHoloTrace);
        
        holoTraces{1 ,tt} = sortedHoloTrace;
        if isfield(ExpStruct,'inputsLP')
            tempHoloTrace_LP{hh,1}=tracedataWin_LP;
            holoTraces_LP{1 ,tt} = sortedHoloTrace_LP;
            sortedholoTracesAllTrials_LP{tt} = vertcat(sortedHoloTrace_LP);
        end
    end
    
    % collect all the spot response across all trials, ordered according to sequence
    sortedholoTracesAllTrials = vertcat(sortedholoTracesAllTrials{:}); %un-nest one level 
    if isfield(ExpStruct,'inputsLP')
        sortedholoTracesAllTrials_LP = vertcat(sortedholoTracesAllTrials_LP{:});
    end
    
    for hh = 1: length(holoSequence)
        datawinsSortedByHolosPerPower{1, hh} = horzcat(sortedholoTracesAllTrials{:, hh});
        if isfield(ExpStruct,'inputsLP')
            datawinsSortedByHolosPerPower_LP{1, hh} = horzcat(sortedholoTracesAllTrials_LP{:, hh});
        end
    end
    
    datawinsSortedByHolosAllPowers{pp,1} = vertcat(datawinsSortedByHolosPerPower);
    wholeTracesSortedByHolos{1, pp} = holoTraces;
    if isfield(ExpStruct,'inputsLP')
        datawinsSortedByHolosAllPowers_LP{pp,1} = vertcat(datawinsSortedByHolosPerPower_LP);
        wholeTracesSortedByHolos_LP{1, pp} = holoTraces_LP;
    end
end
%datawinsSortedByHolosAllPowers = vertcat(datawinsSortedByHolosAllPowers{:});

%%

% rewriting the holo structure in trials x time matrix

repeats = ExpStruct.expParams.repeats; % number of repetitions per power
num_holos_per_power = ExpStruct.holoStimParams.nHolos(1,:); % array of length nConditions
num_pulses_per_holo = this_seq_nPulses;

 % add one to to post + pre length tp account for frame when stim comes on
pscs = zeros(repeats * sum(num_holos_per_power) * num_pulses_per_holo, post_stim_len + pre_stim_len + 1);
psps = zeros(repeats * sum(num_holos_per_power) * num_pulses_per_holo, post_stim_len + pre_stim_len + 1);
stimulus_matrix = zeros(size(ExpStruct.holoRequest.targets,1), repeats * sum(num_holos_per_power) * num_pulses_per_holo); 

% stimulus_matrix should be array of size (num_neurons x num_stim)
% each column shows the power delivered to each neuron on that stim.
% For example, if we're doing 2 spot mapping and we have 6 total cells,
% the first column might look like:
% [0 0 30 0 30 0]
% meaning that we stimmed cells 3 and 5 on the first stim at 30mW.

n=0;

for i = 1:nConditions
    this_power = ExpStruct.holoStimParams.powers(i)*1000;
    
    roi_index = 0;
    for j = 1:ExpStruct.holoStimParams.nHolos(1,i) * num_pulses_per_holo
        
        % increment roi_index every time we cycle through all pulses
        % for a given hologram
        if mod(j-1, num_pulses_per_holo) == 0
            roi_index = roi_index + 1;
        end
        
        if alt_roi_structure
            [this_holo_members] = new_rois{roi_index,:}; 
        elseif isfield(ExpStruct.holoRequest,'condRois')
            [this_holo_members] = ExpStruct.holoRequest.condRois{i,1}{roi_index,:}(1,:);  % for old code cells
        else
            [this_holo_members] = (ExpStruct.holoRequest.condHolos{i,1}{roi_index,:}(1,:));
        end
        
        for k = 1:repeats
            n = n+1;
            pscs(n, :) = datawinsSortedByHolosAllPowers{i,:}{1,j}(:,k);
            if isfield(ExpStruct,'inputsLP')
                psps(n, :) = datawinsSortedByHolosAllPowers_LP{i,:}{1,j}(:,k);
            end
            stimulus_matrix(this_holo_members,n)=this_power;            
            
        end
        
    end
end

% [MARCUS]
% Crop data structures
zero_locs = find(sum(pscs, 2) == 0);
if size(zero_locs, 1) > 0
    zero_start = zero_locs(1);
    pscs = pscs(1:zero_start, :);
    psps = psps(1:zero_start, :);
    stimulus_matrix = stimulus_matrix(:, 1:zero_start);
end

targets = ExpStruct.holoRequest.targets;

if isfield(ExpStruct.holoRequest, 'stack')
    img = ExpStruct.holoRequest.stack;
else
    img = NaN;
end

if isfield(ExpStruct,'pipete_position')
    pipette_position = ExpStruct.pipete_position;
else
    pipette_position = NaN;
end

save(savepath, 'pscs', 'psps', 'stimulus_matrix', 'targets', 'img', 'pipette_position', 'num_pulses_per_holo', '-v7.3');

end
%  figure(1)           
%  imagesc(trials_matrix);hold on; % laser stim
%  xline(100, 'r');
%  
%  xline(100+20*ExpStruct.outParams.pulseDur(1,1), 'r');% depending on duration of laser stim the 2nd red line marks end of laser stim 
%  colorbar;
%  
%  
%  figure(2)
%  imagesc(stimuli_matrix); colorbar;
%  
%  figure(3)
%  imagesc(LP_trials_matrix);hold on; % laser stim
%  xline(100, 'r');
%  
%  xline(100+20*ExpStruct.outParams.pulseDur(1,1), 'r');% depending on duration of laser stim the 2nd red line marks end of laser stim 
%  colorbar;



