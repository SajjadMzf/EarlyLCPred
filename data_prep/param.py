
DATASET = "HIGHD"

UNBALANCED = False

SEQ_LEN = 35
OBS_LEN = 10
PRED_LEN = 25
FPS = 5

grid_max_x = 100
# Rendering params
save_whole_imgs = False
save_cropped_imgs = False
image_scaleH = 4
image_scaleW = 1

if DATASET == 'HIGHD':
    cropped_height = int(20 * image_scaleH)
    cropped_width = int(200 * image_scaleW)
    MAX_VELOCITY_X = 50

# Following list indicate location of virtual vehicle w.r.t. the TV.
# [pv_id, fv_id, rv_id, rpv_id, rfv_id, lv_id, lpv_id, lfv_id],
LARGE_LONGITUDINAL_DIST = 100
LARGE_LATERAL_LANE_DIST = 10
GHOST_VEH1 = [[-1*LARGE_LONGITUDINAL_DIST, 0], 
                [LARGE_LONGITUDINAL_DIST, 0], 
                [0, LARGE_LATERAL_LANE_DIST], 
                [-1*LARGE_LONGITUDINAL_DIST, LARGE_LATERAL_LANE_DIST], 
                [LARGE_LONGITUDINAL_DIST, LARGE_LATERAL_LANE_DIST],
                [0, -1*LARGE_LATERAL_LANE_DIST], 
                [-1*LARGE_LONGITUDINAL_DIST, -1*LARGE_LATERAL_LANE_DIST], 
                [LARGE_LONGITUDINAL_DIST, -1*LARGE_LATERAL_LANE_DIST]]

GHOST_VEH2 = [[LARGE_LONGITUDINAL_DIST, 0], 
                [-1*LARGE_LONGITUDINAL_DIST, 0], 
                [0, LARGE_LATERAL_LANE_DIST], 
                [LARGE_LONGITUDINAL_DIST, LARGE_LATERAL_LANE_DIST], 
                [-1*LARGE_LONGITUDINAL_DIST, LARGE_LATERAL_LANE_DIST],
                [0, -1*LARGE_LATERAL_LANE_DIST], 
                [LARGE_LONGITUDINAL_DIST, -1*LARGE_LATERAL_LANE_DIST], 
                [-1*LARGE_LONGITUDINAL_DIST, -1*LARGE_LATERAL_LANE_DIST]]

SV__LANE_IND1 = [0, 0, -1, -1, -1, 1, 1, 1]
SV__LANE_IND2 = [0, 0, 1, 1, 1, -1, -1, -1]




def generate_paths(first_leg, start_ind, end_ind, second_leg):
    path_list = []
    for i in range(start_ind, end_ind):
        path_list.append(first_leg + str(i+1).zfill(2) + second_leg)
    return path_list

if DATASET == "HIGHD":
    start_ind = 0
    file_numbers = 60
    track_paths = generate_paths('../../Dataset/HighD/Tracks/', 0, 60, '_tracks.csv')
    frame_pickle_paths = generate_paths('../../Dataset/HighD/Pickles/', 0,  60, '_frames.csv')
    track_pickle_paths = generate_paths('../../Dataset/HighD/Pickles/', 0,  60, '_tracks.csv')
    meta_paths = generate_paths('../../Dataset/HighD/Metas/', 0,  60, '_recordingMeta.csv')
    static_paths = generate_paths('../../Dataset/HighD/Statics/', 0,  60, '_tracksMeta.csv')



