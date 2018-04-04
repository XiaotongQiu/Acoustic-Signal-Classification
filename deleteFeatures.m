clc;
dir_to_test='data_23_44_verified';

featuresToDelete='blackman_sub_band_1K_15_dict_unwgt';
data_path=struct('path',{ ...
                        strcat(('/media/885C28DA5C28C532/sound_data/sounds_again/'),dir_to_test,'/nature_daytime'), ... 
                        strcat(('/media/885C28DA5C28C532/sound_data/sounds_again/'),dir_to_test,'/inside_vehicle'), ...
                        strcat(('/media/885C28DA5C28C532/sound_data/sounds_again/'),dir_to_test,'/restaurant'), ...    
                        strcat(('/media/885C28DA5C28C532/sound_data/sounds_again/'),dir_to_test,'/casino'), ...         
                        strcat(('/media/885C28DA5C28C532/sound_data/sounds_again/'),dir_to_test,'/nature_night'), ... 
                        strcat(('/media/885C28DA5C28C532/sound_data/sounds_again/'),dir_to_test,'/bell'), ...         
                        strcat(('/media/885C28DA5C28C532/sound_data/sounds_again/'),dir_to_test,'/playgrounds'), ...  
                        strcat(('/media/885C28DA5C28C532/sound_data/sounds_again/'),dir_to_test,'/street-traffic'), ...         
                        strcat(('/media/885C28DA5C28C532/sound_data/sounds_again/'),dir_to_test,'/thunder'), ...
                        strcat(('/media/885C28DA5C28C532/sound_data/sounds_again/'),dir_to_test,'/train'), ... 
                        strcat(('/media/885C28DA5C28C532/sound_data/sounds_again/'),dir_to_test,'/rain'), ...
                        strcat(('/media/885C28DA5C28C532/sound_data/sounds_again/'),dir_to_test,'/stream'), ...                                               
                        strcat(('/media/885C28DA5C28C532/sound_data/sounds_again/'),dir_to_test,'/ocean'), ...         
                        strcat(('/media/885C28DA5C28C532/sound_data/sounds_again/'),dir_to_test,'/ambulance')} ...                                                                        
                );
            
            
total_dir=size(data_path,2);

for dir_count=1:total_dir
    cd (data_path(dir_count).path);
    all_wav=dir('*.wav');
    
    for wav_index=1:length(all_wav)    
        new_dir=strcat(data_path(dir_count).path,'/',num2str(wav_index));
        cd(new_dir);
        toDel=strcat(new_dir,'/',featuresToDelete);        
        [status, mess,id]=rmdir(toDel,'s');
                
        if(status==0)
            fprintf('%s\n',mess);
        else
            fprintf('Deleted folder in %s \n',toDel);
        end
               
    end
end