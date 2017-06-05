function main(dataset_dir, output_dir)
    % Find all PCG recordings in the validation dataset
    command = ['find "' , dataset_dir , '" -name "*.wav"'];
    [status,list]=system(command);
    if status
        keyboard
        error(['`Find` command failed:', list])
    end
    PCG_Files = strread(list, '%s', 'delimiter', sprintf('\n'));
    %% Load the trained parameter matrices for Springer's HSMM model.
    % The parameters were trained using 409 heart sounds from MIT heart
    % sound database, i.e., recordings a0001-a0409.
    load('Springer_B_matrix.mat');
    load('Springer_pi_vector.mat');
    load('Springer_total_obs_distribution.mat');

    % Load default options for classification
    springer_options   = default_Springer_HSMM_options;

    % Generate s1 and s2 heart sound segmentations for each recording
    for i=1:numel(PCG_Files)
        PCGPath = char(PCG_Files(i));
        disp(['Segmenting file: ', PCGPath])
        % Read in PCG data
        [PCG, Fs1] = audioread(PCGPath);  % load data

        % Resample data
        PCG_resampled = resample(PCG,springer_options.audio_Fs,Fs1); % resample to springer_options.audio_Fs (1000 Hz)

        % Running runSpringerSegmentationAlgorithm.m to obtain the
        % assigned_states
        [segmentations, heartRate] = runSpringerSegmentationAlgorithm(PCG_resampled, springer_options.audio_Fs, Springer_B_matrix, Springer_pi_vector, Springer_total_obs_distribution, false); % obtain the locations for S1, systole, s2 and diastole

        % Plot segmentation against resampled PCG data
        % plot(PCG_resampled);
        % hold on;
        % plot(segmentations);
        keyboard

        % Find changes in beat type classification
        changeIndx = find(abs(diff(segmentations)))+1;
        % Get type of new beat classification (1 = first heart sound (lub), 2 = 2nd
        % heart sound (dub)
        segVals = segmentations(changeIndx);

        assigned_states = segmentations;
        %% We just assume that the assigned_states cover at least 2 whole heart beat cycle
        indx = find(abs(diff(assigned_states))>0); % find the locations with changed states

        if assigned_states(1)>0   % for some recordings, there are state zeros at the beginning of assigned_states
            switch assigned_states(1)
                case 4
                    K=1;
                case 3
                    K=2;
                case 2
                    K=3;
                case 1
                    K=4;
            end
        else
            switch assigned_states(indx(1)+1)
                case 4
                    K=1;
                case 3
                    K=2;
                case 2
                    K=3;
                case 1
                    K=0;
            end
            K=K+1;
        end

        indx2                = indx(K:end);
        rem                  = mod(length(indx2),4);
        indx2(end-rem+1:end) = [];
        A                    = reshape(indx2,4,length(indx2)/4)';

        keyboard
        % Save data to CSV files
        csvdata = [changeIndx, segVals];
        [~, basename, ~] = fileparts(PCGPath);
        csvpath = fullfile(output_dir, strcat(basename, '_segs.csv'));
        dlmwrite (csvpath, [Fs1, springer_options.audio_Fs, heartRate])
        dlmwrite (csvpath, csvdata, '-append')
    end
end

