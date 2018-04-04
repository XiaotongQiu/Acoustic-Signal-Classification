clear all;
clc;

tic

% This is where the htk codes like creating MFCC files, creeting MP
% features are. They also contain config files.

addpath /media/885C28DA5C28C532/Dropbox/code/htk/
save_log_file_at='/media/885C28DA5C28C532/Dropbox/code/htk/logFile.mat';



data_path=struct('path',{ ...
                        '/media/885C28DA5C28C532/Env_sound_interference/0dB/nature_daytime', ... 
                        '/media/885C28DA5C28C532/Env_sound_interference/0dB/inside_vehicle', ...
                        '/media/885C28DA5C28C532/Env_sound_interference/0dB/restaurant', ...    
                        '/media/885C28DA5C28C532/Env_sound_interference/0dB/casino', ...         
                        '/media/885C28DA5C28C532/Env_sound_interference/0dB/nature_night', ... 
                        '/media/885C28DA5C28C532/Env_sound_interference/0dB/bell', ...         
                        '/media/885C28DA5C28C532/Env_sound_interference/0dB/playgrounds', ...  
                        '/media/885C28DA5C28C532/Env_sound_interference/0dB/street-traffic', ...         
                        '/media/885C28DA5C28C532/Env_sound_interference/0dB/thunder', ...
                        '/media/885C28DA5C28C532/Env_sound_interference/0dB/train', ... 
                        '/media/885C28DA5C28C532/Env_sound_interference/0dB/rain', ...
                        '/media/885C28DA5C28C532/Env_sound_interference/0dB/stream', ...                                               
                        '/media/885C28DA5C28C532/Env_sound_interference/0dB/ocean', ...         
                        '/media/885C28DA5C28C532/Env_sound_interference/0dB/ambulance'} ...                                                                        
                );
            
 
           
 
l=logger;            
total_dir=size(data_path,2);

% new_mfcc_dir='unwgt_mfcc_normal_gabor';
% new_mfcc_dir1='wgt_mfcc_normal_gabor';

segment_length=4; %in seconds
no_bits_per_sample=16;
path='./';
segments_var='segments.mat'; % This is where the segments of each set are stored

% Change the directory name if need be
wgt=true;


% If you are changing the sampling rate change the  cuttoff frequency of
% the low pass filter also and of course the downsampling rate
final_sampling_rate=44100;

%% Downsampling
% Before downsampling do a lowpass filter fs was 44.1K and downsampling to
% 22.05K. Therefore design a lowpass filter with cutoff as 0.5pi and then
% do the downsampling
% [fil_B,fil_A]=butter(10,0.5); %using a 10th order butterworth filter
total_sample_per_segment=segment_length*final_sampling_rate;

%% Dictionary
% should contain the dictionary atoms as well as the freq and scale
% information about each atom
 load /media/885C28DA5C28C532/Dropbox/gabor_dict_256x1120.mat

new_mfcc_dir='unwgt_gabor_normal';
new_mfcc_dir1='wgt_gabor_normal';

for dir_count=1:total_dir
    
    disp(strcat('Started ',data_path(dir_count).path))
    l.logMesg(strcat('Starting directory ',data_path(dir_count).path));

% Change the path to where the wav files are stored
% cd /media/885C28DA5C28C532/sound_data/street-traffic/
cd (data_path(dir_count).path);


%OMP code in mex format 
% addpath /media/885C28DA5C28C532/Dropbox/code/compressedSensing/OMPBox

all_wav=dir('*.wav');
save 'file_order' all_wav;

%%
for wav_index=1:length(all_wav)
    
    
    
    new_dir=strcat(path,num2str(wav_index));
    mkdir(new_dir);
    data=wavread(all_wav(wav_index).name);
    
    %take only monochannel
    data_wanted=data(:,1);
    
%     Clear data to make save some RAM space
    data=[];
    
    %% Try without downsampling!
    %filter the signal to avoid aliasing before downsampling
%     data_wanted=filter(fil_B,fil_A,data_wanted);
%     data_wanted=downsample(data_wanted,2); % Downsample by a factor of 2 to get the needed 
    
    % Downsample to the required sampling rate
%     wavwrite(data_wanted,final_sampling_rate,strcat(new_dir,'/',num2str(wav_index),'.wav')); % each sample here is represented by 16 bits    
    
    %Take the downsampled data for all further processing
%     new_data=wavread(strcat(new_dir,'/',num2str(wav_index),'.wav'));
%     delete(strcat(new_dir,'/',num2str(wav_index),'.wav'));
    new_data=data_wanted;
    number_of_segments=floor(length(new_data)/total_sample_per_segment);
    segments=reshape(new_data(1:number_of_segments*total_sample_per_segment),[],total_sample_per_segment);
      
    cd (new_dir);
    save 'segments' segments 
    
    for seg_index=1:size(segments,1)
           wavwrite(segments(seg_index,:),final_sampling_rate,strcat(num2str(seg_index),'.wav')); % each sample here is represented by 16 bits    
    end
    
    
               
           %% Method 1 - Creating a SCP file - will have difficulty in extracting the wav signals
           %            Shell scripting to generate MFCCs\

%             Get all wav files in the directory
%            ! ls *.wav > wav_file  
% %            Create a scp file
%            ! sed s/\.wav/\.mfc/g wav_file >mfc_file
%            ! paste wav_file mfc_file >final.scp
%            ! rm wav_file
%            ! rm mfc_file      
% %            get mfcc for these wav files
%            !~/softwares/htk-3.4/HTKTools/HCopy -C  /media/885C28DA5C28C532/Dropbox/code/htk/HCopyConfig  -S final.scp
%%         Method 2 - Create MFCC file directly using wav files            
            !for f in *.wav; do  /home/sunit/softwares/htk-3.4/HTKTools/HCopy -C  /media/885C28DA5C28C532/Dropbox/code/htk/HCopyConfig28_44 $f  ${f//.\wav/\.mfc} > ${f//.\wav/\.txt}; done
            %% Modify the MFCC to suit requirements
%             if(wgt)
%                 if(isdir('wgt_mfcc'))
%                     rmdir wgt_mfcc
%                 end
%                 mkdir wgt_mfcc;
%                 new_mfcc_dir='wgt_mfcc';
%             else
%                 if(isdir('unwgt_mfcc'))
%                     rmdir unwgt_mfcc
%                 end
%                 
%                 mkdir unwgt_mfcc;
%                 new_mfcc_dir='unwgt_mfcc';
%             end
            
%% TODO : check if the directory already exists and take neccesssary action

%%
          

            mkdir(new_mfcc_dir);
            
            if(exist('new_mfcc_dir1','var'))
                mkdir(new_mfcc_dir1);
                createAllMFCCs(pwd,wgt,new_mfcc_dir,dict,frequency,scale,new_mfcc_dir1);
            else
                createAllMFCCs(pwd,wgt,new_mfcc_dir1,dict,frequency,scale);
            end
    cd ..;          

    l.logMesg(strcat('Completing and exiting directory ',new_mfcc_dir));
    save(save_log_file_at,'l');

end
 l.logMesg(strcat('Exiting directory ',data_path(dir_count).path));
 disp(strcat('Exiting ',data_path(dir_count).path))
 save(save_log_file_at,'l');
end

time_spent=toc;
l.logMesg(strcat('time spent ',num2str(time_spent)))
save(save_log_file_at,'l');