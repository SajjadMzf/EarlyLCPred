import os
import time
from multiprocessing import Process
import numpy as np 

from extract_scenarios import ExtractScenarios
from render_scenarios import RenderScenarios
import param as p


def extract(core_num, file_row, track_path, pickle_path_frame, pickle_path_tracks, meta_path, static_path, dataset_name):
    np.random.seed(0)
    if isinstance(track_path, str):
        i = 0
        file_num = file_row + i + 1
        print('Core number: ', core_num, ' started on file number: ', file_num)
        start = time.time()
        extractor = ExtractScenarios(
            file_num,
            track_path, 
            pickle_path_tracks,
            pickle_path_frame, 
            static_path,
            meta_path,
            dataset_name)
        extractor.extract_and_save()
        
            
        end = time.time()
        print('Core Number: ', core_num, ' Ended in: ', end-start, ' s.')
    else:
        start = time.time()
        for i in range(len(track_path)):
            file_num = file_row + i + 1
            print('Core number: ', core_num, ' started on file number: ', file_num)
            
            extractor = ExtractScenarios(
                file_num,
                track_path[i], 
                pickle_path_tracks[i],
                pickle_path_frame[i], 
                static_path[i],
                meta_path[i],
                dataset_name)
            extractor.extract_and_save() 
        
        end = time.time()
        print('Core Number: ', core_num, ' Ended in: ', end-start, ' s.')

def render(core_num, file_row, track_path, pickle_path_frame, pickle_path_tracks, meta_path, static_path, dataset_name):
    np.random.seed(0)
    if isinstance(track_path, str):
        i = 0
        file_num = file_row + i + 1
        print('Core number: ', core_num, ' started on file number: ', file_num)
        start = time.time()
        renderer = RenderScenarios(
            file_num,
            track_path, 
            pickle_path_frame, 
            static_path,
            meta_path,
            dataset_name
        ) 
        renderer.load_scenarios()
        renderer.render_scenarios()
        renderer.save_dataset() 
        end = time.time()
        print('Core Number: ', core_num, ' Ended in: ', end-start, ' s.')
    else:
        for i in range(len(track_path)):
            file_num = file_row + i + 1
            print('Core number: ', core_num, ' started on file number: ', file_num)
            start = time.time()
            
            renderer = RenderScenarios(
                file_num,
                track_path[i], 
                pickle_path_frame[i], 
                static_path[i],
                meta_path[i],
                dataset_name
            )
            renderer.update_dirs() 
            renderer.load_scenarios()
            renderer.render_scenarios()
            renderer.save_dataset()  

            end = time.time()
        print('Core Number: ', core_num, ' Ended in: ', end-start, ' s.')




if __name__ =="__main__":
    
    
    if  p.DATASET != "HIGHD":
        for i in range(p.file_numbers):
            extract(i, i, p.track_paths[i], p.frame_pickle_paths[i], p.track_pickle_paths[i], p.meta_paths[i], p.static_paths[i], p.DATASET)
            render(i, i, p.track_paths[i], p.frame_pickle_paths[i], p.track_pickle_paths[i], p.meta_paths[i], p.static_paths[i], p.DATASET)
        
    else:
        
        # Single Core (For Debugging purposes)
        '''
        i = 26
        #extract(i, i, p.track_paths[i], p.frame_pickle_paths[i], p.track_pickle_paths[i], p.meta_paths[i], p.static_paths[i],  p.DATASET)    
        render(i, i, p.track_paths[i], p.frame_pickle_paths[i], p.track_pickle_paths[i], p.meta_paths[i], p.static_paths[i],  p.DATASET)
        exit()
        '''

        # Extract LC Scenarios 
        total_cores = 3
        total_files = p.file_numbers - p.start_ind
        file_per_core = int(total_files/total_cores)
        procs = []
        for core_num in range(total_cores):
            file_row = file_per_core*core_num + p.start_ind
            
            proc = Process(target= extract, args = ( 
                                                core_num+1,
                                                file_row, 
                                                p.track_paths[file_row:(file_row+file_per_core)], 
                                                p.frame_pickle_paths[file_row:(file_row+file_per_core)], 
                                                p.track_pickle_paths[file_row:(file_row+file_per_core)],
                                                p.meta_paths[file_row:(file_row+file_per_core)], 
                                                p.static_paths[file_row:(file_row+file_per_core)],
                                                p.DATASET))
            procs.append(proc)
            proc.start()
        
        for proc in procs:
            proc.join()

        # Render Whole Dataset Using Multi-threads
        for core_num in range(total_cores):
            file_row = file_per_core*core_num + p.start_ind
            
            proc = Process(target= render, args = ( 
                                                core_num+1,
                                                file_row, 
                                                p.track_paths[file_row:(file_row+file_per_core)], 
                                                p.frame_pickle_paths[file_row:(file_row+file_per_core)], 
                                                p.track_pickle_paths[file_row:(file_row+file_per_core)],
                                                p.meta_paths[file_row:(file_row+file_per_core)], 
                                                p.static_paths[file_row:(file_row+file_per_core)],
                                                p.DATASET))
            procs.append(proc)
            proc.start()
        
        for proc in procs:
            proc.join()