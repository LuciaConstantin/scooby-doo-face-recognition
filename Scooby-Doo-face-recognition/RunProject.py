from Parameters import *
from FacialDetector import *
import pdb
from Visualize import *

# set parameters for rectangle window

params: Parameters = Parameters()

# dreptunghi 72x88 pixeli
params.dim_window_w = 72 
params.dim_window_h = 88
params.dim_hog_cell_rect = 4  

params.overlap_rect = 0.3

params.number_positive_examples_rect = 3144 
params.number_negative_examples_rect = 8000

params.threshold_rect = 1 


# patrat 40x40 pixeli
params.dim_window_sq= 40 
params.dim_hog_cell_sq = 4  

params.overlap_rect = 0.3

params.number_positive_examples_sq = 3236 
params.number_negative_examples_sq = 8000

params.threshold_sq = 3


# patrat 64x96 pixeli
params.dim_window_inv_w = 64
params.dim_window_inv_h = 96
params.dim_hog_cell_inv = 4  

params.overlap_inv = 0.3

params.number_positive_examples_inv = 3584 
params.number_negative_examples_inv = 8000

params.threshold_inv = 1

params.has_annotations = False
params.use_flip_images = True

if params.use_flip_images:
    params.number_positive_examples_sq *= 2
    params.number_positive_examples_rect *=2
    params.number_positive_examples_inv *=2
    

facial_detector: FacialDetector = FacialDetector(params)


# exemple pozitive

# rectangle
positive_features_path_rect = os.path.join(params.dir_save_files, 'descriptoriExemplePozitiveDreptunghi_' + str(params.dim_hog_cell_rect) + '_' +
                        str(params.number_positive_examples_rect) + '.npy')

if os.path.exists(positive_features_path_rect):
    positive_features_rect = np.load(positive_features_path_rect)
    print('Am incarcat descriptorii pentru exemplele pozitive dreptunghi 72x88')
else:
    print('Construim descriptorii pentru exemplele pozitive dreptunghi:')
    positive_features_rect = facial_detector.get_positive_descriptors_rect()
    np.save(positive_features_path_rect, positive_features_rect)
    print('Am salvat descriptorii pentru exemplele pozitive dreptunghi in fisierul %s' % positive_features_path_rect)
    
# square
positive_features_path_sq = os.path.join(params.dir_save_files, 'descriptoriExemplePozitivePatrat_' + str(params.dim_hog_cell_sq) + '_' +
                        str(params.number_positive_examples_sq) + '.npy')
if os.path.exists(positive_features_path_sq):
    positive_features_sq = np.load(positive_features_path_sq)
    print('Am incarcat descriptorii pentru exemplele pozitive patrat')
else:
    print('Construim descriptorii pentru exemplele pozitive patrat:')
    positive_features_sq = facial_detector.get_positive_descriptors_square()
    np.save(positive_features_path_sq, positive_features_sq)
    print('Am salvat descriptorii pentru exemplele pozitive in fisierul %s' % positive_features_path_sq)


# inverse
positive_features_path_inv = os.path.join(params.dir_save_files, 'descriptoriExemplePozitive_inv_' + str(params.dim_hog_cell_inv) + '_' +
                        str(params.number_positive_examples_inv) + '.npy')

if os.path.exists(positive_features_path_inv):
    positive_features_inv = np.load(positive_features_path_inv)
    print('Am incarcat descriptorii pentru exemplele pozitive dreptunghi 64x96')
else:
    print('Construim descriptorii pentru exemplele pozitive inverse:')
    positive_features_inv = facial_detector.get_positive_descriptors_rect()
    np.save(positive_features_path_inv, positive_features_inv)
    print('Am salvat descriptorii pentru exemplele pozitive dreptunghi in fisierul %s' % positive_features_path_inv)



# exemple negative


# dreptunghi
negative_features_path_rect = os.path.join(params.dir_save_files, 'descriptoriExempleNegativeDreptunghi_' + str(params.dim_hog_cell_rect) + '_' +
                        str(params.number_negative_examples_rect) + '.npy')

if os.path.exists(negative_features_path_rect):
    negative_features_rect = np.load(negative_features_path_rect)
    print('Am incarcat descriptorii pentru exemplele negative dreptunghi 72x88')
else:
    print('Construim descriptorii pentru exemplele negative dreptunghi:')
    negative_features_rect = facial_detector.get_negative_descriptors_rect()
    np.save(negative_features_path_rect, negative_features_rect)
    print('Am salvat descriptorii pentru exemplele negative in fisierul %s' % negative_features_path_rect)
    
# patrat
negative_features_path_sq = os.path.join(params.dir_save_files, 'descriptoriExempleNegativePatrat_' + str(params.dim_hog_cell_sq) + '_' +
                        str(params.number_negative_examples_sq) + '.npy')
if os.path.exists(negative_features_path_sq):
    negative_features_sq = np.load(negative_features_path_sq)
    print('Am incarcat descriptorii pentru exemplele negative patrat')
else:
    print('Construim descriptorii pentru exemplele negative patrat:')
    negative_features_sq = facial_detector.get_negative_descriptors_square()
    np.save(negative_features_path_sq, negative_features_sq)
    print('Am salvat descriptorii pentru exemplele negative in fisierul %s' % negative_features_path_sq)


# inverse
negative_features_path_inv = os.path.join(params.dir_save_files, 'descriptoriExempleNegative_inv_' + str(params.dim_hog_cell_inv) + '_' +
                        str(params.number_negative_examples_inv) + '.npy')

if os.path.exists(negative_features_path_inv):
    negative_features_inv = np.load(negative_features_path_inv)
    print('Am incarcat descriptorii pentru exemplele negative dreptunghi 64x96')
else:
    print('Construim descriptorii pentru exemplele negative inverse:')
    negative_features_inv = facial_detector.get_negative_descriptors_rect()
    np.save(negative_features_path_inv, negative_features_inv)
    print('Am salvat descriptorii pentru exemplele negative in fisierul %s' % negative_features_path_inv)



    
# Pasul 4. Invatam clasificatorul liniar

# dreptunghi

training_examples_rect = np.concatenate((np.squeeze(positive_features_rect), np.squeeze(negative_features_rect)), axis=0)
train_labels_rect = np.concatenate((np.ones(params.number_positive_examples_rect), np.zeros(negative_features_rect.shape[0])))

facial_detector.train_classifier_rect(training_examples_rect, train_labels_rect)

# patrat

training_examples_sq = np.concatenate((np.squeeze(positive_features_sq), np.squeeze(negative_features_sq)), axis=0)
train_labels_sq = np.concatenate((np.ones(params.number_positive_examples_sq), np.zeros(negative_features_sq.shape[0])))

facial_detector.train_classifier_square(training_examples_sq, train_labels_sq)

# inverse

training_examples_inv = np.concatenate((np.squeeze(positive_features_inv), np.squeeze(negative_features_inv)), axis=0)
train_labels_inv = np.concatenate((np.ones(params.number_positive_examples_inv), np.zeros(negative_features_inv.shape[0])))

facial_detector.train_classifier_inverse(training_examples_inv, train_labels_inv)


detections, scores, file_names = facial_detector.run_multiple()


'''
if params.has_annotations:
    facial_detector.eval_detections(detections, scores, file_names)
    show_detections_with_ground_truth(detections, scores, file_names, params)
else:
    show_detections_without_ground_truth(detections, scores, file_names, params)
''' 
    
# Salvare in fisiere    

all_detections = np.array(detections, dtype=int)
all_scores = np.array(scores, dtype=float)
all_file_names = np.array(file_names)


save_dir = "463_Constantin_Lucia/task1"
os.makedirs(save_dir, exist_ok=True)

        
np.save(os.path.join(save_dir, "detections_all_faces.npy"), all_detections)
np.save(os.path.join(save_dir, "scores_all_faces.npy"), all_scores)
np.save(os.path.join(save_dir, "file_names_all_faces.npy"), all_file_names)