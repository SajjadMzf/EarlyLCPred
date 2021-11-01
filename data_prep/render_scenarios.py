import os
import cv2
import numpy as np 
import pickle
import h5py
import matplotlib.pyplot as plt
import read_csv as rc
import param as p
from utils import rendering_funcs as rf

class RenderScenarios:
    """This class is for rendering extracted scenarios from HighD dataset recording files (needs to be called seperately for each scenario).
    """
    def __init__(
        self,
        file_num:'Number of recording file being rendered',
        track_path:'Path to track file', 
        pickle_path:'Path to pickle file', 
        static_path:'Path to static file',
        meta_path:'Path to meta file',
        dataset_name: 'Dataset  Name'):
        self.seq_len = p.SEQ_LEN
        self.metas = rc.read_meta_info(meta_path)
        self.fr_div = self.metas[rc.FRAME_RATE]/p.FPS
        self.track_path = track_path
        self.scenarios = []
        self.file_num = file_num
        
        # Default Settings
        self.save_whole_imgs = p.save_whole_imgs
        self.save_cropped_imgs = p.save_cropped_imgs
        
        ''' 1. Representation Properties:'''
        self.filled = True
        self.empty = False
        self.dtype = bool
        # 1.1 Scales and distances
        self.image_scaleW = p.image_scaleW
        self.image_scaleH = p.image_scaleH
        self.lines_width = 1
        self.dash_lines = tuple([8,8])
        self.highway_top_margin = int(5 * self.image_scaleH)
        self.highway_bottom_margin = int(5 * self.image_scaleH)
        self.cropped_height = p.cropped_height
        self.cropped_width = p.cropped_width
        # 1.3 Others
        self.mid_barrier = False
        
        if p.UNBALANCED:
            dir_ext = 'U'
        else:
            dir_ext = ''
       
        self.LC_whole_imgs_rdir = "../../Dataset/" + dataset_name + "/NOEV_WholeImages" + dir_ext
        self.LC_cropped_imgs_rdir = "../../Dataset/" + dataset_name + "/NOEV_CroppedImages" + dir_ext
        
        self.LC_full_states_rdir = "../../Dataset/" + dataset_name + "/NOEV_FullStates" + dir_ext
        self.LC_states_dir = "../../Dataset/" + dataset_name + "/NOEV_States" + dir_ext 
        self.LC_states_dataset_rdir = "../../Dataset/" + dataset_name + "/NOEV_StateDataset" + dir_ext
        self.LC_image_dataset_rdir = "../../Dataset/" + dataset_name + "/NOEV_ImageDataset" + dir_ext
         
        self.frames_data, image_width = rc.read_track_csv(track_path, pickle_path, group_by = 'frames', fr_div = self.fr_div)
        self.statics = rc.read_static_info(static_path)
        self.orig_image_width = image_width
        self.image_width = int(image_width * self.image_scaleW)
        self.image_height = int((self.metas[rc.LOWER_LANE_MARKINGS][-1])*self.image_scaleH + self.highway_top_margin + self.highway_bottom_margin)
        self.update_dirs()
        
    def load_scenarios(self):
        file_dir = os.path.join(self.LC_states_dir, str(self.file_num).zfill(2) + '.pickle')
        with open(file_dir, 'rb') as handle:
            self.scenarios = pickle.load(handle)
    
    def save_full_scenarios_dict(self):
        file_dir = os.path.join(self.LC_full_states_dir, str(self.file_num).zfill(2) + '.pickle')
        with open(file_dir, 'wb') as handle:
            pickle.dump(self.scenarios, handle, protocol= pickle.HIGHEST_PROTOCOL)
    
    def save_dataset(self):
        file_dir = os.path.join(self.LC_image_dataset_dir, str(self.file_num).zfill(2) + '.h5')
        npy_dir = os.path.join(self.LC_image_dataset_dir, str(self.file_num).zfill(2) + '.npy')
        hf = h5py.File(file_dir, 'w')
        valid_itrs = [False if scenario['images'] is None else True for scenario in self.scenarios]
        data_num = valid_itrs.count(True)
        image_data = hf.create_dataset('image_data', shape = (data_num, self.seq_len, 3, self.cropped_height, self.cropped_width), dtype = np.bool)
        frame_data = hf.create_dataset('frame_data', shape = (data_num, self.seq_len), dtype = np.float32)       
        tv_data = hf.create_dataset('tv_data', shape = (data_num,), dtype = np.int)
        labels = hf.create_dataset('labels', shape = (data_num,), dtype = np.float32)
        state_data = hf.create_dataset('state_data', shape = (data_num, self.seq_len, 18), dtype = np.float32)
        ttlc_available = hf.create_dataset('ttlc_available', shape = (data_num,), dtype = np.bool)
        

        data_itr = 0
        for itr, validity in enumerate(valid_itrs):
            if validity == False:
                continue
            temp = self.scenarios[itr]['images']
            temp = np.transpose(temp,[0,3,1,2])# Chanel first
            image_data[data_itr, :] = temp
            state_data[data_itr, :] = self.scenarios[itr]['states']
            frame_data[data_itr, :] = self.scenarios[itr]['frames']
            tv_data[data_itr] = self.scenarios[itr]['tv']
            labels[data_itr] = self.scenarios[itr]['label']
            ttlc_available[data_itr] = self.scenarios[itr]['ttlc_available']
            data_itr += 1
        hf.close()
        np.save(npy_dir, data_itr) 
    
      

        

    def render_scenarios(self)-> "Number of rendered and saved scenarios":
        saved_data_number = 0
        
        for scenario_idx, scenario in enumerate(self.scenarios):
            
            tv_id = scenario['tv']
            label = scenario['label']
            driving_dir = scenario['driving_dir']

            scene_cropped_imgs = []
            whole_imgs = []
            img_frames = []
            scene_cav_ids = []
            svs_vises = []
            states = []
            tv_lane_ind = None
            number_of_fr = self.seq_len 
            for fr in range(number_of_fr):
                frame = scenario['frames'][fr]
                
                svs_ids = scenario['svs']['id'][:,fr]
                cropped_img, whole_img, valid, state, tv_lane_ind = self.plot_frame(
                    self.frames_data[int(frame/self.fr_div -1)],
                    tv_id, 
                    svs_ids,
                    driving_dir,
                    frame,
                    tv_lane_ind
                    )
                
                # Being valid is about width of TV being less than 2 pixels
                if not valid:
                    print('Invalid frame:', fr+1, ' of scenario: ', scenario_idx+1, ' of ', len(self.scenarios))
                    break
                scene_cropped_imgs.append(cropped_img)
                whole_imgs.append(whole_img)
                img_frames.append(frame)
                states.append(state)
                
            if not valid:
                continue
            
            scene_cropped_imgs = np.array(scene_cropped_imgs, dtype = self.dtype)
            self.scenarios[scenario_idx]['images'] = scene_cropped_imgs
            self.scenarios[scenario_idx]['states'] = np.array(states)
            saved_data_number += 1
            
            if self.save_whole_imgs: rf.save_image_sequence( tv_id, img_frames, whole_imgs, os.path.join(self.LC_whole_imgs_dir, str(label)), self.file_num)
            if self.save_cropped_imgs: rf.save_image_sequence( tv_id, img_frames, scene_cropped_imgs, os.path.join(self.LC_cropped_imgs_dir, str(label)), self.file_num)
            
        return saved_data_number

    def plot_frame(
        self, 
        frame_data:'Data array of current frame', 
        tv_id:'ID of the TV', 
        svs_ids:'IDs of the SVs', 
        driving_dir:'TV driving direction',
        frame:'frame',
        tv_lane_ind:'The TV lane index of its initial frame'):
        
        veh_channel, lane_channel, obs_channel = rf.initialize_representation(self.image_width, self.image_height, rep_dtype = self.dtype, filled_value = self.filled, occlusion= False)
        assert(frame_data[rc.FRAME]==frame)   
        tv_itr = np.nonzero(frame_data[rc.TRACK_ID] == tv_id)[0][0]
        svs_itr = np.array([np.nonzero(frame_data[rc.TRACK_ID] == sv_id)[0][0] if sv_id!=0 else None for sv_id in svs_ids])
        
        vehicle_in_frame_number = len(frame_data[rc.TRACK_ID])
        
        # Lambda function for calculating vehicles x,y, width and length
        corner_x = lambda itr: int(frame_data[rc.X][itr]*self.image_scaleW)
        corner_y = lambda itr: int((frame_data[rc.Y][itr])*self.image_scaleH) + self.highway_top_margin
        veh_width = lambda itr: int(frame_data[rc.WIDTH][itr]*self.image_scaleW)
        veh_height = lambda itr: int(frame_data[rc.HEIGHT][itr]*self.image_scaleH)
        center_x = lambda itr: int(frame_data[rc.X][itr]*self.image_scaleW + veh_width(itr)/2)
        center_y = lambda itr: int(frame_data[rc.Y][itr]*self.image_scaleH + veh_height(itr)/2)  + self.highway_top_margin
        
        

        
        for itr in range(vehicle_in_frame_number):
            veh_channel = rf.draw_vehicle(veh_channel, corner_x(itr), corner_y(itr), veh_width(itr), veh_height(itr), self.filled)
        

        # If a barrier is conisdered at the middle of highway blocking the view, the image height will change depending on driving direction
        if self.mid_barrier:
            mid_barrier = int(((self.metas[rc.UPPER_LANE_MARKINGS][-1] + self.metas[rc.LOWER_LANE_MARKINGS][0])/2) * self.image_scaleH)  + self.highway_top_margin
            dir_image_height = lambda driving_dir: [0, mid_barrier-2] if driving_dir==1 else [mid_barrier+2, self.image_height]
        else:
            dir_image_height = lambda driving_dir: [0, self.image_height]
        
        # Draw lines
        lane_channel = rf.draw_lane_markings(lane_channel, 
                                self.image_width, 
                                (self.metas[rc.LOWER_LANE_MARKINGS])* self.image_scaleH  + self.highway_top_margin,
                                (self.metas[rc.UPPER_LANE_MARKINGS])* self.image_scaleH  + self.highway_top_margin,
                                self.lines_width, 
                                self.filled, 
                                self.dash_lines*self.image_scaleW
                                )
        
        # Crop image
        image = np.concatenate((veh_channel, lane_channel, obs_channel), axis = 2)
        
        
        tv_lane_markings = (self.metas[rc.UPPER_LANE_MARKINGS])* self.image_scaleH  + self.highway_top_margin if driving_dir == 1 else (self.metas[rc.LOWER_LANE_MARKINGS])* self.image_scaleH + self.highway_top_margin
        tv_lane_markings = tv_lane_markings.astype(int)
        if tv_lane_ind is None:
            tv_lane_ind = 0
            for ind, value in reversed(list(enumerate(tv_lane_markings))):
                if center_y(tv_itr)>value:
                    tv_lane_ind = ind
                    break
        
        cropped_img, valid = rf.crop_image(image, 
                                    self.image_width, 
                                    self.image_height,
                                    int(center_x(tv_itr)),
                                    int(center_y(tv_itr)),
                                    tv_lane_markings,
                                    driving_dir,
                                    tv_lane_ind,
                                    self.cropped_height,
                                    self.cropped_width,
                                    self.lines_width,
                                    self.filled)
        
        state = np.zeros((18)) # From Wirthmuller 2021
        
        def clamp(n, minn, maxn):
            if n < minn:
                return minn
            elif n > maxn:
                return maxn
            else:
                return n

        if tv_lane_ind+1>=len(tv_lane_markings):# len is 1-based lane_ind is 0-based
            return cropped_img, image, False, [], [], [], []
        
        lane_width = (tv_lane_markings[tv_lane_ind+1]-tv_lane_markings[tv_lane_ind])
        
        #velocity_x = lambda itr: abs(frame_data[rc.X_VELOCITY][itr])/p.MAX_VELOCITY_X
        fix_sign = lambda x: x if driving_dir == 1 else -1*x

        tv_left_lane_ind = tv_lane_ind + 1 if driving_dir==1 else tv_lane_ind

        lateral_pos = lambda itr, lane_ind: abs(frame_data[rc.Y][itr] + frame_data[rc.HEIGHT][itr]/2- tv_lane_markings[lane_ind])

        rel_distance_x = lambda itr: fix_sign(frame_data[rc.X][itr] - frame_data[rc.X][tv_itr])

        rel_distance_y = lambda itr: fix_sign(frame_data[rc.Y][itr] - frame_data[rc.Y][tv_itr])
        
        rel_velo_x = lambda itr: fix_sign(frame_data[rc.X_VELOCITY][itr] - frame_data[rc.X_VELOCITY][tv_itr]) #transform from [-1,1] to [0,1]
        rel_velo_y =lambda itr: fix_sign(frame_data[rc.Y_VELOCITY][itr] - frame_data[rc.Y_VELOCITY][tv_itr])
        rel_acc_x = lambda itr: fix_sign(frame_data[rc.X_ACCELERATION][itr] - frame_data[rc.X_ACCELERATION][tv_itr])
        rel_acc_y =lambda itr: fix_sign(frame_data[rc.Y_ACCELERATION][itr] - frame_data[rc.Y_ACCELERATION][tv_itr])

        
        #state[0]: existance of left lane
        if (tv_lane_ind+2==len(tv_lane_markings) and driving_dir == 1) or (tv_lane_ind ==0 and driving_dir==2):
            state[0] = 0
        else:
            state[0] = 1
        # state[1]: existance of right lane
        if (tv_lane_ind+2==len(tv_lane_markings) and driving_dir == 2) or (tv_lane_ind ==0 and driving_dir==1):
            state[1] = 0
        else:
            state[1] = 1
        
        state[2] = lane_width # lane width

        # svs : [pv_id, fv_id, rv_id, rpv_id, rfv_id, lv_id, lpv_id, lfv_id]
        pv_itr = svs_itr[0]
        fv_itr = svs_itr[1]
        rv_itr = svs_itr[2]
        rpv_itr = svs_itr[3]
        rfv_itr = svs_itr[4]
        lv_itr = svs_itr[5]
        lpv_itr = svs_itr[6]
        lfv_itr = svs_itr[7]
        

        # long dist to pv
        state[3] = rel_distance_x(pv_itr) if pv_itr != None else 100 
        # long dist to rpv
        state[4] = rel_distance_x(rpv_itr) if rpv_itr != None else 100 
        # long dist to fv
        state[5] = rel_distance_x(fv_itr) if fv_itr != None else -100 
        # lat dist to left marking
        state[6] = lateral_pos(tv_itr, tv_left_lane_ind)
         # lat dist to right vehicle
        state[7] = rel_distance_y(rv_itr) if rv_itr != None else -2*lane_width
        # lat dist to rfv
        state[8] =rel_distance_y(rfv_itr) if rfv_itr != None else -2*lane_width
        # rel long vel pv
        state[9] = rel_velo_x(pv_itr) if pv_itr != None else 0
        # rel long vel fv
        state[10] = rel_velo_x(fv_itr) if fv_itr != None else 0
        # rel lat vel pv
        state[11] = rel_velo_y(pv_itr) if pv_itr != None else 0
        # rel lat vel rpv
        state[12] = rel_velo_y(rpv_itr) if rpv_itr != None else 0
        #  rel lat vel lv
        state[13] = rel_velo_y(lv_itr) if lv_itr != None else 0
        # rel lat vel rv
        state[14] = rel_velo_y(rv_itr) if rv_itr != None else 0
         # long acc of tv
        state[15] = frame_data[rc.X_ACCELERATION][tv_itr]
        # rpv rel long acc
        state[16] = rel_acc_x(rpv_itr) if rpv_itr != None else 0
        # lat acc of tv
        state[17] = frame_data[rc.Y_ACCELERATION][tv_itr]
        
        return cropped_img, image, valid, state, tv_lane_ind
        

    
    def update_dirs(self):
        
        self.dataset_version_dir = 'NOEV'
        
        self.LC_cropped_imgs_dir = self.LC_cropped_imgs_rdir
        if not os.path.exists(self.LC_cropped_imgs_dir):
            os.makedirs(self.LC_cropped_imgs_dir)
        
        for i in range(3):
            label_dir = os.path.join(self.LC_cropped_imgs_dir, str(i))
            if not os.path.exists(label_dir):
                os.makedirs(label_dir) 

        self.LC_whole_imgs_dir = self.LC_whole_imgs_rdir
        if not os.path.exists(self.LC_whole_imgs_dir):
            os.makedirs(self.LC_whole_imgs_dir)
    
        for i in range(3):
            label_dir = os.path.join(self.LC_whole_imgs_dir, str(i))
            if not os.path.exists(label_dir):
                os.makedirs(label_dir) 
            
        self.LC_full_states_dir = self.LC_full_states_rdir
        if not os.path.exists(self.LC_full_states_dir):
            os.makedirs(self.LC_full_states_dir)

        self.LC_states_dataset_dir = self.LC_states_dataset_rdir
        if not os.path.exists(self.LC_states_dataset_dir):
            os.makedirs(self.LC_states_dataset_dir)
        
        self.LC_image_dataset_dir = self.LC_image_dataset_rdir
        if not os.path.exists(self.LC_image_dataset_dir):
            os.makedirs(self.LC_image_dataset_dir)
        
    