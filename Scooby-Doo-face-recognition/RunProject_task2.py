from Parameters_task2 import *
from CharacterDetector import *
import pdb
from Visualize import *


params: Parameters_task2 = Parameters_task2()

params.dim_window_w = 72 # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
params.dim_window_h = 88
params.dim_hog_cell = 4  # dimensiunea celulei


params.overlap = 0.3


params.number_positive_examples_shaggy = 697  
params.number_negative_examples_shaggy = 6588  
        
params.number_positive_examples_daphne = 662 
params.number_negative_examples_daphne = 7900 
        
params.number_positive_examples_fred = 855 
params.number_negative_examples_fred = 6428
        
params.number_positive_examples_velma = 1054  
params.number_negative_examples_velma = 6217 



params.threshold = -2 # toate ferestrele cu scorul > threshold si maxime locale devin detectii  era 4.5 inainte
params.has_annotations = False

params.use_hard_mining = False  # (optional)antrenare cu exemple puternic negative
params.use_flip_images = True  # adauga imaginile cu fete oglindite

if params.use_flip_images:
    
    params.number_positive_examples_daphne *= 2
    params.number_positive_examples_fred *=2
    params.number_positive_examples_shaggy *=2
    params.number_positive_examples_velma *=2
    

character_detector: CharacterDetector = CharacterDetector(params)

# Pasii 1+2+3. Incarcam exemplele pozitive (cropate) si exemple negative generate
# verificam daca sunt deja existente



# velma

positive_features_path = os.path.join(params.dir_save_files, 'descriptoriExemplePozitiveVelma_' + str(params.dim_hog_cell) + '_' +
                        str(params.number_positive_examples_velma) + '.npy')
if os.path.exists(positive_features_path):
    positive_features_velma = np.load(positive_features_path)
    print('Am incarcat descriptorii pentru exemplele pozitive')
else:
    print('Construim descriptorii pentru exemplele pozitive:')
    positive_features_velma = character_detector.get_positive_descriptors()
    np.save(positive_features_path, positive_features_velma)
    print('Am salvat descriptorii pentru exemplele pozitive in fisierul %s' % positive_features_path)


#fred
positive_features_path = os.path.join(params.dir_save_files, 'descriptoriExemplePozitiveFred_' + str(params.dim_hog_cell) + '_' +
                        str(params.number_positive_examples_fred) + '.npy')
if os.path.exists(positive_features_path):
    positive_features_fred = np.load(positive_features_path)
    print('Am incarcat descriptorii pentru exemplele pozitive')
else:
    print('Construim descriptorii pentru exemplele pozitive:')
    positive_features_fred = character_detector.get_positive_descriptors()
    np.save(positive_features_path, positive_features_fred)
    print('Am salvat descriptorii pentru exemplele pozitive in fisierul %s' % positive_features_path)



# shaggy
positive_features_path = os.path.join(params.dir_save_files, 'descriptoriExemplePozitiveShaggy_' + str(params.dim_hog_cell) + '_' +
                        str(params.number_positive_examples_shaggy) + '.npy')
if os.path.exists(positive_features_path):
    positive_features_shaggy = np.load(positive_features_path)
    print('Am incarcat descriptorii pentru exemplele pozitive')
else:
    print('Construim descriptorii pentru exemplele pozitive:')
    positive_features_shaggy = character_detector.get_positive_descriptors()
    np.save(positive_features_path, positive_features_shaggy)
    print('Am salvat descriptorii pentru exemplele pozitive in fisierul %s' % positive_features_path)


# daphne

positive_features_path = os.path.join(params.dir_save_files, 'descriptoriExemplePozitiveDaphne_' + str(params.dim_hog_cell) + '_' +
                        str(params.number_positive_examples_daphne) + '.npy')
if os.path.exists(positive_features_path):
    positive_features_daphne = np.load(positive_features_path)
    print('Am incarcat descriptorii pentru exemplele pozitive')
else:
    print('Construim descriptorii pentru exemplele pozitive:')
    positive_features_daphne = character_detector.get_positive_descriptors()
    np.save(positive_features_path, positive_features_daphne)
    print('Am salvat descriptorii pentru exemplele pozitive in fisierul %s' % positive_features_path)

# exemple negative

# velma

negative_features_path = os.path.join(params.dir_save_files, 'descriptoriExempleNegativeVelma_' + str(params.dim_hog_cell) + '_' +
                        str(params.number_negative_examples_velma) + '.npy')
if os.path.exists(negative_features_path):
    negative_features_velma = np.load(negative_features_path)
    print('Am incarcat descriptorii pentru exemplele negative')
else:
    print('Construim descriptorii pentru exemplele negative:')
    negative_features_velma = character_detector.get_negative_descriptors_new()
    np.save(negative_features_path, negative_features_velma)
    print('Am salvat descriptorii pentru exemplele negative in fisierul %s' % negative_features_path)


# fred
negative_features_path = os.path.join(params.dir_save_files, 'descriptoriExempleNegativeFred_' + str(params.dim_hog_cell) + '_' +
                        str(params.number_negative_examples_fred) + '.npy')
if os.path.exists(negative_features_path):
    negative_features_fred = np.load(negative_features_path)
    print('Am incarcat descriptorii pentru exemplele negative')
else:
    print('Construim descriptorii pentru exemplele negative:')
    negative_features_fred = character_detector.get_negative_descriptors_new()
    np.save(negative_features_path, negative_features_fred)
    print('Am salvat descriptorii pentru exemplele negative in fisierul %s' % negative_features_path)




# daphne
negative_features_path = os.path.join(params.dir_save_files, 'descriptoriExempleNegativeDaphne_' + str(params.dim_hog_cell) + '_' +
                        str(params.number_negative_examples_daphne) + '.npy')
if os.path.exists(negative_features_path):
    negative_features_daphne = np.load(negative_features_path)
    print('Am incarcat descriptorii pentru exemplele negative')
else:
    print('Construim descriptorii pentru exemplele negative:')
    negative_features_daphne = character_detector.get_negative_descriptors_new()
    np.save(negative_features_path, negative_features_daphne)
    print('Am salvat descriptorii pentru exemplele negative in fisierul %s' % negative_features_path)
    


# shaggy
negative_features_path = os.path.join(params.dir_save_files, 'descriptoriExempleNegativeShaggy_' + str(params.dim_hog_cell) + '_' +
                        str(params.number_negative_examples_shaggy) + '.npy')
if os.path.exists(negative_features_path):
    negative_features_shaggy = np.load(negative_features_path)
    print('Am incarcat descriptorii pentru exemplele negative')
else:
    print('Construim descriptorii pentru exemplele negative:')
    negative_features_shaggy = character_detector.get_negative_descriptors_new()
    np.save(negative_features_path, negative_features_shaggy)
    print('Am salvat descriptorii pentru exemplele negative in fisierul %s' % negative_features_path)



# Pasul 4. Invatam clasificatorul liniar
# daphne
training_examples_daphne = np.concatenate((np.squeeze(positive_features_daphne), np.squeeze(negative_features_daphne)), axis=0)
train_labels_daphne = np.concatenate((np.ones(params.number_positive_examples_daphne), np.zeros(negative_features_daphne.shape[0])))

character_detector.train_classifier_daphne(training_examples_daphne, train_labels_daphne)

# fred
training_examples_fred = np.concatenate((np.squeeze(positive_features_fred), np.squeeze(negative_features_fred)), axis=0)
train_labels_fred = np.concatenate((np.ones(params.number_positive_examples_fred), np.zeros(negative_features_fred.shape[0])))

character_detector.train_classifier_fred(training_examples_fred, train_labels_fred)

# shaggy

training_examples_shaggy = np.concatenate((np.squeeze(positive_features_shaggy), np.squeeze(negative_features_shaggy)), axis=0)
train_labels_shaggy = np.concatenate((np.ones(params.number_positive_examples_shaggy), np.zeros(negative_features_shaggy.shape[0])))

character_detector.train_classifier_shaggy(training_examples_shaggy, train_labels_shaggy)


# velma


training_examples_velma = np.concatenate((np.squeeze(positive_features_velma), np.squeeze(negative_features_velma)), axis=0)
train_labels_velma = np.concatenate((np.ones(params.number_positive_examples_velma), np.zeros(negative_features_velma.shape[0])))

character_detector.train_classifier_velma(training_examples_velma, train_labels_velma)



save_dir = "463_Constantin_Lucia/task2"
os.makedirs(save_dir, exist_ok=True)

detections, scores, file_names = character_detector.run_recognition(params.path_task1_detections, "shaggy")

        
np.save(os.path.join(save_dir, "detections_shaggy.npy"), detections)
np.save(os.path.join(save_dir, "scores_shaggy.npy"), scores)
np.save(os.path.join(save_dir, "file_names_shaggy.npy"), file_names)



detections, scores, file_names = character_detector.run_recognition(params.path_task1_detections, "daphne")
        
np.save(os.path.join(save_dir, "detections_daphne.npy"), detections)
np.save(os.path.join(save_dir, "scores_daphne.npy"), scores)
np.save(os.path.join(save_dir, "file_names_daphne.npy"), file_names)



detections, scores, file_names = character_detector.run_recognition(params.path_task1_detections, "velma")
        
np.save(os.path.join(save_dir, "detections_velma.npy"), detections)
np.save(os.path.join(save_dir, "scores_velma.npy"), scores)
np.save(os.path.join(save_dir, "file_names_velma.npy"), file_names)

detections, scores, file_names = character_detector.run_recognition(params.path_task1_detections, "fred")
        
np.save(os.path.join(save_dir, "detections_fred.npy"), detections)
np.save(os.path.join(save_dir, "scores_fred.npy"), scores)
np.save(os.path.join(save_dir, "file_names_fred.npy"), file_names)








'''
if params.has_annotations:
    character_detector.eval_detections_character(detections, scores, file_names, 'shaggy')
    
    show_detections_with_ground_truth(detections, scores, file_names, params)
    
else:
    show_detections_without_ground_truth(detections, scores, file_names, params)
    
'''