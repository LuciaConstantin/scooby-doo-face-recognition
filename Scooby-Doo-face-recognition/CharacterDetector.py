from Parameters_task2 import *
import numpy as np
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import glob
import cv2 as cv
import pdb
import pickle
import ntpath
from copy import deepcopy
import timeit
from skimage.feature import hog


class CharacterDetector:
    def __init__(self, params:Parameters_task2):
        self.params = params
        self.best_model_daphne = None
        self.best_model_fred = None
        self.best_model_shaggy = None
        self.best_model_velma = None
        

    def get_positive_descriptors(self):
        # in aceasta functie calculam descriptorii pozitivi
        # vom returna un numpy array de dimensiuni NXD
        # unde N - numar exemplelor pozitive
        # iar D - dimensiunea descriptorului
        # D = (params.dim_window/params.dim_hog_cell - 1) ^ 2 * params.dim_descriptor_cell (fetele sunt patrate)

        images_path = os.path.join(self.params.dir_pos_examples, '*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        positive_descriptors = []
        print('Calculam descriptorii pt %d imagini pozitive...' % num_images)
        for i in range(num_images):
            print('Procesam exemplul pozitiv numarul %d...' % i)
            img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)
         
            features = hog(img, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                           cells_per_block=(2, 2), feature_vector=True)
            print(len(features))

            positive_descriptors.append(features)
            if self.params.use_flip_images:
                features = hog(np.fliplr(img), pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                               cells_per_block=(2, 2), feature_vector=True)
                positive_descriptors.append(features)

        positive_descriptors = np.array(positive_descriptors)
        return positive_descriptors

   
    
    def get_negative_descriptors_new(self):
    
        images_path = os.path.join(self.params.dir_neg_examples, '*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        negative_descriptors = []

        print('Calculam descriptorii pentru %d imagini negative...' % num_images)

        for i, file in enumerate(files):
            print('Procesam exemplul negativ numarul %d...' % i)
            img = cv.imread(file, cv.IMREAD_GRAYSCALE)

            features = hog(img,
                        pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                        cells_per_block=(2, 2),
                        feature_vector=True)
            negative_descriptors.append(features)
            
        negative_descriptors = np.array(negative_descriptors)
        return negative_descriptors
    

    def train_classifier_fred(self, training_examples, train_labels):
        
        svm_file_name = os.path.join(self.params.dir_save_files, 'best_model_fred_%d_%d_%d' %
                                        (self.params.dim_hog_cell, self.params.number_negative_examples_fred,
                                        self.params.number_positive_examples_fred))
        
        if os.path.exists(svm_file_name):
            self.best_model_fred = pickle.load(open(svm_file_name, 'rb'))
            return

        best_accuracy = 0
        best_c = 0
        best_model_fred = None
        Cs = [10 ** -5, 10 ** -4,  10 ** -3,  10 ** -2, 10 ** -1, 10 ** 0]
        for c in Cs:
            print('Antrenam un clasificator pentru c=%f' % c)
            model = LinearSVC(C=c)
            model.fit(training_examples, train_labels)
            acc = model.score(training_examples, train_labels)
            print(acc)
            if acc > best_accuracy:
                best_accuracy = acc
                best_c = c
                best_model_fred = deepcopy(model)

        print('Performanta clasificatorului optim pt c = %f' % best_c)
        # salveaza clasificatorul
        pickle.dump(best_model_fred, open(svm_file_name, 'wb'))

        # vizualizeaza cat de bine sunt separate exemplele pozitive de cele negative dupa antrenare
        # ideal ar fi ca exemplele pozitive sa primeasca scoruri > 0, iar exemplele negative sa primeasca scoruri < 0
        scores = best_model_fred.decision_function(training_examples)
        self.best_model_fred = best_model_fred
        positive_scores = scores[train_labels > 0]
        negative_scores = scores[train_labels <= 0]


        plt.plot(np.sort(positive_scores))
        plt.plot(np.zeros(len(positive_scores)))
        plt.plot(np.sort(negative_scores))
        plt.xlabel('Nr example antrenare')
        plt.ylabel('Scor clasificator')
        plt.title('Distributia scorurilor clasificatorului pe exemplele de antrenare')
        plt.legend(['Scoruri exemple pozitive', '0', 'Scoruri exemple negative'])
        plt.show()

    def train_classifier_velma(self, training_examples, train_labels):
    
        svm_file_name = os.path.join(self.params.dir_save_files, 'best_model_velma_%d_%d_%d' %
                                        (self.params.dim_hog_cell, self.params.number_negative_examples_velma,
                                        self.params.number_positive_examples_velma))
        
        if os.path.exists(svm_file_name):
            self.best_model_velma = pickle.load(open(svm_file_name, 'rb'))
            return

        best_accuracy = 0
        best_c = 0
        best_model_velma = None
        Cs = [10 ** -5, 10 ** -4,  10 ** -3,  10 ** -2, 10 ** -1, 10 ** 0]
        for c in Cs:
            print('Antrenam un clasificator pentru c=%f' % c)
            model = LinearSVC(C=c)
            model.fit(training_examples, train_labels)
            acc = model.score(training_examples, train_labels)
            print(acc)
            if acc > best_accuracy:
                best_accuracy = acc
                best_c = c
                best_model_velma = deepcopy(model)

        print('Performanta clasificatorului optim pt c = %f' % best_c)
        # salveaza clasificatorul
        pickle.dump(best_model_velma, open(svm_file_name, 'wb'))

        # vizualizeaza cat de bine sunt separate exemplele pozitive de cele negative dupa antrenare
        # ideal ar fi ca exemplele pozitive sa primeasca scoruri > 0, iar exemplele negative sa primeasca scoruri < 0
        scores = best_model_velma.decision_function(training_examples)
        self.best_model_velma = best_model_velma
        positive_scores = scores[train_labels > 0]
        negative_scores = scores[train_labels <= 0]


        plt.plot(np.sort(positive_scores))
        plt.plot(np.zeros(len(positive_scores)))
        plt.plot(np.sort(negative_scores))
        plt.xlabel('Nr example antrenare')
        plt.ylabel('Scor clasificator')
        plt.title('Distributia scorurilor clasificatorului pe exemplele de antrenare')
        plt.legend(['Scoruri exemple pozitive', '0', 'Scoruri exemple negative'])
        plt.show()
    
    def train_classifier_daphne(self, training_examples, train_labels):
        
        svm_file_name = os.path.join(self.params.dir_save_files, 'best_model_daphne_%d_%d_%d' %
                                        (self.params.dim_hog_cell, self.params.number_negative_examples_daphne,
                                        self.params.number_positive_examples_daphne))
        
        if os.path.exists(svm_file_name):
            self.best_model_daphne = pickle.load(open(svm_file_name, 'rb'))
            return

        best_accuracy = 0
        best_c = 0
        best_model_daphne= None
        Cs = [10 ** -5, 10 ** -4,  10 ** -3,  10 ** -2, 10 ** -1, 10 ** 0]
        for c in Cs:
            print('Antrenam un clasificator pentru c=%f' % c)
            model = LinearSVC(C=c)
            model.fit(training_examples, train_labels)
            acc = model.score(training_examples, train_labels)
            print(acc)
            if acc > best_accuracy:
                best_accuracy = acc
                best_c = c
                best_model_daphne = deepcopy(model)

        print('Performanta clasificatorului optim pt c = %f' % best_c)
        # salveaza clasificatorul
        pickle.dump(best_model_daphne, open(svm_file_name, 'wb'))

        # vizualizeaza cat de bine sunt separate exemplele pozitive de cele negative dupa antrenare
        # ideal ar fi ca exemplele pozitive sa primeasca scoruri > 0, iar exemplele negative sa primeasca scoruri < 0
        scores = best_model_daphne.decision_function(training_examples)
        self.best_model_daphne = best_model_daphne
        positive_scores = scores[train_labels > 0]
        negative_scores = scores[train_labels <= 0]


        plt.plot(np.sort(positive_scores))
        plt.plot(np.zeros(len(positive_scores)))
        plt.plot(np.sort(negative_scores))
        plt.xlabel('Nr example antrenare')
        plt.ylabel('Scor clasificator')
        plt.title('Distributia scorurilor clasificatorului pe exemplele de antrenare')
        plt.legend(['Scoruri exemple pozitive', '0', 'Scoruri exemple negative'])
        plt.show()

    def train_classifier_shaggy(self, training_examples, train_labels):
        
        svm_file_name = os.path.join(self.params.dir_save_files, 'best_model_shaggy_%d_%d_%d' %
                                        (self.params.dim_hog_cell, self.params.number_negative_examples_shaggy,
                                        self.params.number_positive_examples_shaggy))
        
        
        if os.path.exists(svm_file_name):
            self.best_model_shaggy = pickle.load(open(svm_file_name, 'rb'))
            return

        best_accuracy = 0
        best_c = 0
        best_model_shaggy = None
        Cs = [10 ** -5, 10 ** -4,  10 ** -3,  10 ** -2, 10 ** -1, 10 ** 0]
        for c in Cs:
            print('Antrenam un clasificator pentru c=%f' % c)
            model = LinearSVC(C=c)
            model.fit(training_examples, train_labels)
            acc = model.score(training_examples, train_labels)
            print(acc)
            if acc > best_accuracy:
                best_accuracy = acc
                best_c = c
                best_model_shaggy = deepcopy(model)

        print('Performanta clasificatorului optim pt c = %f' % best_c)
        # salveaza clasificatorul
        pickle.dump(best_model_shaggy, open(svm_file_name, 'wb'))

        # vizualizeaza cat de bine sunt separate exemplele pozitive de cele negative dupa antrenare
        # ideal ar fi ca exemplele pozitive sa primeasca scoruri > 0, iar exemplele negative sa primeasca scoruri < 0
        scores = best_model_shaggy.decision_function(training_examples)
        self.best_model_shaggy = best_model_shaggy
        positive_scores = scores[train_labels > 0]
        negative_scores = scores[train_labels <= 0]


        plt.plot(np.sort(positive_scores))
        plt.plot(np.zeros(len(positive_scores)))
        plt.plot(np.sort(negative_scores))
        plt.xlabel('Nr example antrenare')
        plt.ylabel('Scor clasificator')
        plt.title('Distributia scorurilor clasificatorului pe exemplele de antrenare')
        plt.legend(['Scoruri exemple pozitive', '0', 'Scoruri exemple negative'])
        plt.show()

        
        
    def non_maximal_suppression(self, image_detections, image_scores, image_size):
        """
        Detectiile cu scor mare suprima detectiile ce se suprapun cu acestea dar au scor mai mic.
        Detectiile se pot suprapune partial, dar centrul unei detectii nu poate
        fi in interiorul celeilalte detectii.
        :param image_detections:  numpy array de dimensiune NX4, unde N este numarul de detectii.
        :param image_scores: numpy array de dimensiune N
        :param image_size: tuplu, dimensiunea imaginii
        :return: image_detections si image_scores care sunt maximale.
        """

        # xmin, ymin, xmax, ymax
        x_out_of_bounds = np.where(image_detections[:, 2] > image_size[1])[0]
        y_out_of_bounds = np.where(image_detections[:, 3] > image_size[0])[0]
        print(x_out_of_bounds, y_out_of_bounds)
        image_detections[x_out_of_bounds, 2] = image_size[1]
        image_detections[y_out_of_bounds, 3] = image_size[0]
        sorted_indices = np.flipud(np.argsort(image_scores))
        sorted_image_detections = image_detections[sorted_indices]
        sorted_scores = image_scores[sorted_indices]

        is_maximal = np.ones(len(image_detections)).astype(bool)
        iou_threshold = 0.3
        for i in range(len(sorted_image_detections) - 1):
            if is_maximal[i] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                for j in range(i + 1, len(sorted_image_detections)):
                    if is_maximal[j] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                        if self.intersection_over_union(sorted_image_detections[i],sorted_image_detections[j]) > iou_threshold:is_maximal[j] = False
                        else:  # verificam daca centrul detectiei este in mijlocul detectiei cu scor mai mare
                            c_x = (sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2
                            c_y = (sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2
                            if sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2] and \
                                    sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3]:
                                is_maximal[j] = False
        return sorted_image_detections[is_maximal], sorted_scores[is_maximal]
        


    def run_recognition(self, task1_folder, character):
        detections = []
        scores = []
        file_names = []

        w = {
            "daphne": self.best_model_daphne.coef_.T,
            "fred": self.best_model_fred.coef_.T,
            "shaggy": self.best_model_shaggy.coef_.T,
            "velma": self.best_model_velma.coef_.T
        }

        b = {
            "daphne": self.best_model_daphne.intercept_[0],
            "fred": self.best_model_fred.intercept_[0],
            "shaggy": self.best_model_shaggy.intercept_[0],
            "velma": self.best_model_velma.intercept_[0]
        }

        
        detections_task1 = np.load(
            os.path.join(task1_folder, "detections_all_faces.npy")
        )
        file_names_task1 = np.load(
            os.path.join(task1_folder, "file_names_all_faces.npy")
        )

        
        for det, img_name in zip(detections_task1, file_names_task1):
            x_min, y_min, x_max, y_max = det.astype(int)

            img = cv.imread(
                os.path.join(self.params.dir_test_examples, img_name),
                cv.IMREAD_GRAYSCALE
            )
            if img is None:
                continue

            h, w_img = img.shape

            if character == 'shaggy':
                pond_x, pond_y = 0.20, 0.10
            elif character == 'daphne':
                pond_x, pond_y = 0.30, 0.25
            elif character == 'velma':
                pond_x, pond_y = 0.20, 0.10
            elif character == 'fred':
                pond_x, pond_y = 0.20, 0.30
           
            pad_x = int(pond_x * (x_max - x_min))
            pad_y = int(pond_y * (y_max - y_min))

            x_min_p = max(0, x_min - pad_x)
            y_min_p = max(0, y_min - pad_y)
            x_max_p = min(w_img, x_max + pad_x)
            y_max_p = min(h, y_max + pad_y)

            face = img[y_min_p:y_max_p, x_min_p:x_max_p]
            if face.size == 0:
                continue

            face = cv.resize(
                face,
                (self.params.dim_window_w, self.params.dim_window_h)
            )

            descr = hog(
                face,
                pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                cells_per_block=(2, 2),
                feature_vector=True
            )

            score = np.dot(descr, w[character])[0] + b[character]

            if score >= self.params.threshold:
                detections.append([x_min, y_min, x_max, y_max])
                scores.append(score)
                file_names.append(img_name)

        return (
            np.array(detections),
            np.array(scores),
            np.array(file_names)
        )




    
    def compute_average_precision(self, rec, prec):
        # functie adaptata din 2010 Pascal VOC development kit
        m_rec = np.concatenate(([0], rec, [1]))
        m_pre = np.concatenate(([0], prec, [0]))
        for i in range(len(m_pre) - 1, -1, 1):
            m_pre[i] = max(m_pre[i], m_pre[i + 1])
        m_rec = np.array(m_rec)
        i = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
        average_precision = np.sum((m_rec[i] - m_rec[i - 1]) * m_pre[i])
        return average_precision
    
    

    def intersection_over_union(self, bbox_a, bbox_b):
        x_a = max(bbox_a[0], bbox_b[0])
        y_a = max(bbox_a[1], bbox_b[1])
        x_b = min(bbox_a[2], bbox_b[2])
        y_b = min(bbox_a[3], bbox_b[3])

        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

        box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
        box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

        iou = inter_area / float(box_a_area + box_b_area - inter_area)

        return iou

    def eval_detections_character(self,detections, scores, file_names,character):
        ground_truth_file = np.loadtxt(self.params.path_annotations, dtype='str')
        ground_truth_file_names = np.array(ground_truth_file[:, 0])
        ground_truth_detections = np.array(ground_truth_file[:, 1:], int)

        num_gt_detections = len(ground_truth_detections)  # numar total de adevarat pozitive
        gt_exists_detection = np.zeros(num_gt_detections)
        # sorteazam detectiile dupa scorul lor
        sorted_indices = np.argsort(scores)[::-1]
        file_names = file_names[sorted_indices]
        scores = scores[sorted_indices]
        detections = detections[sorted_indices]

        num_detections = len(detections)
        true_positive = np.zeros(num_detections)
        false_positive = np.zeros(num_detections)
        duplicated_detections = np.zeros(num_detections)

        for detection_idx in range(num_detections):
            indices_detections_on_image = np.where(ground_truth_file_names == file_names[detection_idx])[0]

            gt_detections_on_image = ground_truth_detections[indices_detections_on_image]
            bbox = detections[detection_idx]
            max_overlap = -1
            index_max_overlap_bbox = -1
            for gt_idx, gt_bbox in enumerate(gt_detections_on_image):
                overlap = self.intersection_over_union(bbox, gt_bbox)
                if overlap > max_overlap:
                    max_overlap = overlap
                    index_max_overlap_bbox = indices_detections_on_image[gt_idx]

            # clasifica o detectie ca fiind adevarat pozitiva / fals pozitiva
            if max_overlap >= 0.3:
                if gt_exists_detection[index_max_overlap_bbox] == 0:
                    true_positive[detection_idx] = 1
                    gt_exists_detection[index_max_overlap_bbox] = 1
                else:
                    false_positive[detection_idx] = 1
                    duplicated_detections[detection_idx] = 1
            else:
                false_positive[detection_idx] = 1

        cum_false_positive = np.cumsum(false_positive)
        cum_true_positive = np.cumsum(true_positive)

        rec = cum_true_positive / num_gt_detections
        prec = cum_true_positive / (cum_true_positive + cum_false_positive)
        average_precision = self.compute_average_precision(rec, prec)
        plt.plot(rec, prec, '-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(character + ' faces: average precision %.3f' % average_precision)
        plt.savefig('precizie_medie_' + character + '.png')
        plt.show()