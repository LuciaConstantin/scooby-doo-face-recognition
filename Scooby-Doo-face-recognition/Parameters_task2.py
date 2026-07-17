import os

class Parameters_task2:
    def __init__(self):
        self.base_dir = ''
        self.dir_pos_examples = os.path.join(self.base_dir, 'daphne_resized72x88_pad')
        self.dir_neg_examples = os.path.join(self.base_dir, 'negative_daphne_72x88_pad')
        #self.dir_test_examples = os.path.join(self.base_dir,'validare/validare')  # 'exempleTest/CursVA'   'exempleTest/CMU+MIT'
        self.dir_test_examples = os.path.join(self.base_dir,'C:/Users/Lucia/Desktop/tema2-final/validare/validare')
        
        self.path_annotations = os.path.join(self.base_dir, 'validare/task2_shaggy_gt_validare.txt')
        self.path_task1_detections = os.path.join(self.base_dir, '463_Constantin_Lucia/task1')
        
        
        self.dir_save_files = os.path.join(self.base_dir, 'salveazaFisiere')
        if not os.path.exists(self.dir_save_files):
            os.makedirs(self.dir_save_files)
            print('directory created: {} '.format(self.dir_save_files))
        else:
            print('directory {} exists '.format(self.dir_save_files))

        # set the parameters
        self.dim_window_w = 72
        self.dim_window_h = 88
        self.dim_hog_cell = 4 
        self.dim_descriptor_cell = 72
        self.overlap = 0.3
        self.has_annotations = False
        self.threshold = 0
        self.use_flip_images = True


        self.number_positive_examples_shaggy = 697  
        self.number_negative_examples_shaggy = 6588  
        
        
        self.number_positive_examples_daphne = 662 
        self.number_negative_examples_daphne = 7900 
        
        self.number_positive_examples_fred = 855 
        self.number_negative_examples_fred = 6428
        
        self.number_positive_examples_velma = 1054  
        self.number_negative_examples_velma = 6217 

