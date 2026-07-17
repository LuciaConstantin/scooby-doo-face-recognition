import os

class Parameters:
    def __init__(self):
        # files
        self.base_dir = ''
        self.dir_pos_examples_rect = os.path.join(self.base_dir, 'faces_rect') 
        self.dir_neg_examples_rect = os.path.join(self.base_dir, 'negative_examples_rect')
        
        self.dir_pos_examples_square = os.path.join(self.base_dir, 'positive_orig') 
        self.dir_neg_examples_square = os.path.join(self.base_dir, 'negative_examples_orig')
        
        self.dir_test_examples = os.path.join(self.base_dir,'validare')
        self.path_annotations = os.path.join(self.base_dir, 'validare/task1_gt_validare.txt')
        
        
        self.dir_save_files = os.path.join(self.base_dir, 'salvFis')
        
        if not os.path.exists(self.dir_save_files):
            os.makedirs(self.dir_save_files)
            print('directory created: {} '.format(self.dir_save_files))
        else:
            print('directory {} exists '.format(self.dir_save_files))

        # fereastra dreptunghiulara 72x88 pixeli
        self.dim_window_w = 72
        self.dim_window_h = 88
        self.dim_hog_cell_rect = 4 
        self.overlap_rect = 0.3
        self.number_positive_examples_rect = 3144  
        self.number_negative_examples_rect = 8000 
        self.threshold_rect = 0
        
        # fereastra patrata 40x40 pixeli
        self.dim_window_sq= 40
        self.dim_hog_cell_sq = 4 
        self.overlap_sq = 0.3
        self.number_positive_examples_sq = 3236  
        self.number_negative_examples_sq = 8000 
        self.threshold_sq = 0
        
        self.has_annotations = False
        self.use_flip_images = False
        
        
        # fereastra dreptunghiulara 64x96 pixeli
        self.dim_window_inv_w = 64
        self.dim_window_inv_h = 96
        self.dim_hog_cell_inv = 4 
        self.overlap_inv = 0.3
        self.number_positive_examples_inv = 3584  
        self.number_negative_examples_inv = 8000 
        self.threshold_inv = 0
        