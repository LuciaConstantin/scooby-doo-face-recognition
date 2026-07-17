from Parameters import *
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


class FacialDetector:
    def __init__(self, params:Parameters):
        self.params = params
        self.best_model_rect = None
        self.best_model_sq = None
        self.best_model_inv = None
        
        

    def get_positive_descriptors_rect(self):
        images_path = os.path.join(self.params.dir_pos_examples_rect, '*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        positive_descriptors = []
        print('Calculam descriptorii pt %d imagini pozitive...' % num_images)
        for i in range(num_images):
            print('Procesam exemplul pozitiv numarul %d...' % i)
            img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)
         
            features = hog(img, pixels_per_cell=(self.params.dim_hog_cell_rect, self.params.dim_hog_cell_rect),
                           cells_per_block=(2, 2), feature_vector=True)

            positive_descriptors.append(features)
            if self.params.use_flip_images:
                features = hog(np.fliplr(img), pixels_per_cell=(self.params.dim_hog_cell_rect, self.params.dim_hog_cell_rect),
                               cells_per_block=(2, 2), feature_vector=True)
                positive_descriptors.append(features)

        positive_descriptors = np.array(positive_descriptors)
        return positive_descriptors
    
    def get_positive_descriptors_square(self):
        images_path = os.path.join(self.params.dir_pos_examples_square, '*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        positive_descriptors = []
        print('Calculam descriptorii pt %d imagini pozitive...' % num_images)
        for i in range(num_images):
            print('Procesam exemplul pozitiv numarul %d...' % i)
            img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)
         
            features = hog(img, pixels_per_cell=(self.params.dim_hog_cell_sq, self.params.dim_hog_cell_sq),
                           cells_per_block=(2, 2), feature_vector=True)

            positive_descriptors.append(features)
            if self.params.use_flip_images:
                features = hog(np.fliplr(img), pixels_per_cell=(self.params.dim_hog_cell_sq, self.params.dim_hog_cell_sq),
                               cells_per_block=(2, 2), feature_vector=True)
                positive_descriptors.append(features)

        positive_descriptors = np.array(positive_descriptors)
        return positive_descriptors

   
    def get_negative_descriptors_rect(self):
    
        images_path = os.path.join(self.params.dir_neg_examples_rect, '*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        negative_descriptors = []

        print('Calculam descriptorii pentru %d imagini negative...' % num_images)

        for i, file in enumerate(files):
            print('Procesam exemplul negativ numarul %d...' % i)
            img = cv.imread(file, cv.IMREAD_GRAYSCALE)

            features = hog(img,
                        pixels_per_cell=(self.params.dim_hog_cell_rect, self.params.dim_hog_cell_rect),
                        cells_per_block=(2, 2),
                        feature_vector=True)
            negative_descriptors.append(features)
            
        negative_descriptors = np.array(negative_descriptors)
        return negative_descriptors
    
    def get_negative_descriptors_square(self):
    
        images_path = os.path.join(self.params.dir_neg_examples_square, '*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        negative_descriptors = []

        print('Calculam descriptorii pentru %d imagini negative...' % num_images)

        for i, file in enumerate(files):
            print('Procesam exemplul negativ numarul %d...' % i)
            img = cv.imread(file, cv.IMREAD_GRAYSCALE)

            features = hog(img,
                        pixels_per_cell=(self.params.dim_hog_cell_sq, self.params.dim_hog_cell_sq),
                        cells_per_block=(2, 2),
                        feature_vector=True)
            negative_descriptors.append(features)
            
        negative_descriptors = np.array(negative_descriptors)
        return negative_descriptors
    

    def train_classifier_rect(self, training_examples, train_labels):
        print("primul dreptunghi")
        svm_file_name = os.path.join(self.params.dir_save_files, 'best_model_rect_%d_%d_%d' %
                                     (self.params.dim_hog_cell_rect, self.params.number_negative_examples_rect,
                                      self.params.number_positive_examples_rect))
        if os.path.exists(svm_file_name):
            self.best_model_rect = pickle.load(open(svm_file_name, 'rb'))
            return

        best_accuracy = 0
        best_c = 0
        best_model_rect = None
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
                best_model_rect = deepcopy(model)

        print('Performanta clasificatorului optim pt c = %f' % best_c)
        # salveaza clasificatorul
        pickle.dump(best_model_rect, open(svm_file_name, 'wb'))

        scores = best_model_rect.decision_function(training_examples)
        self.best_model_rect = best_model_rect
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
        
        
    def train_classifier_square(self, training_examples, train_labels):
        print("primul patrat")
        
        svm_file_name = os.path.join(self.params.dir_save_files, 'best_model_square_%d_%d_%d' %
                                     (self.params.dim_hog_cell_sq, self.params.number_negative_examples_sq,
                                      self.params.number_positive_examples_sq))
        if os.path.exists(svm_file_name):
            self.best_model_sq = pickle.load(open(svm_file_name, 'rb'))
            return

        best_accuracy = 0
        best_c = 0
        best_model_sq = None
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
                best_model_sq = deepcopy(model)

        print('Performanta clasificatorului optim pt c = %f' % best_c)
        # salveaza clasificatorul
        pickle.dump(best_model_sq, open(svm_file_name, 'wb'))

        scores = best_model_sq.decision_function(training_examples)
        self.best_model_sq = best_model_sq
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
        
        
    def train_classifier_inverse(self, training_examples, train_labels):
        print("al 2 lea dreptunghi")
        
        svm_file_name = os.path.join(self.params.dir_save_files, 'best_model_inv_%d_%d_%d' %
                                     (self.params.dim_hog_cell_inv, self.params.number_negative_examples_inv,
                                      self.params.number_positive_examples_inv))
        if os.path.exists(svm_file_name):
            self.best_model_inv = pickle.load(open(svm_file_name, 'rb'))
            return

        best_accuracy = 0
        best_c = 0
        best_model_inv = None
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
                best_model_inv = deepcopy(model)

        print('Performanta clasificatorului optim pt c = %f' % best_c)
        # salveaza clasificatorul
        pickle.dump(best_model_inv, open(svm_file_name, 'wb'))

        scores = best_model_inv.decision_function(training_examples)
        self.best_model_inv = best_model_inv
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


   
    
    def run_multiple(self):
        sizes = [0.4, 0.5, 0.6, 0.8, 0.9, 1.0, 1.2]
        
        test_images_path = os.path.join(self.params.dir_test_examples, '*.jpg')
        test_files = glob.glob(test_images_path)
        detections = None  
        
        scores = np.array([]) 
        file_names = np.array([])  
       
        w_rect = self.best_model_rect.coef_.T
        bias_rect = self.best_model_rect.intercept_[0]
        
        w_sq = self.best_model_sq.coef_.T
        bias_sq = self.best_model_sq.intercept_[0]
        
        w_inv = self.best_model_inv.coef_.T
        bias_inv = self.best_model_inv.intercept_[0]
        
        
        num_test_images = len(test_files)
       
        
        for i in range(num_test_images):
            start_time = timeit.default_timer()
            print('Procesam imaginea de testare %d/%d..' % (i, num_test_images))
            img = cv.imread(test_files[i], cv.IMREAD_GRAYSCALE)
            
            image_scores_all = []
            image_detections_all = []
            
            for s in sizes:
                width = int(img.shape[1] * s)
                height = int(img.shape[0] * s)
                    
                img_res = cv.resize(img, (width, height))
                
                if s != 0.4 and s != 0.5:
            # dreptunghi 72x88 pixeli
                    image_scores = []
                    image_detections = []
                
                    hog_descriptor_rect = hog(img_res, pixels_per_cell=(self.params.dim_hog_cell_rect, self.params.dim_hog_cell_rect),
                                cells_per_block=(2, 2), feature_vector=False)
                    num_cols = img_res.shape[1]//self.params.dim_hog_cell_rect-1
                    num_rows = img_res.shape[0]//self.params.dim_hog_cell_rect-1
                    num_cel_template_w = self.params.dim_window_w // self.params.dim_hog_cell_rect-1
                    num_cel_template_h = self.params.dim_window_h // self.params.dim_hog_cell_rect-1
                    
                    
                    for y in range(0, num_rows-num_cel_template_h):
                        for x in range(0, num_cols-num_cel_template_w):
                            descr = hog_descriptor_rect[y:y+num_cel_template_h, x:x+num_cel_template_w].flatten()
                            score = np.dot(descr, w_rect)[0] + bias_rect
                            if score > self.params.threshold_rect:
                                s_x = img.shape[1] / width
                                s_y = img.shape[0] / height
                                
                                x_min= int(x*self.params.dim_hog_cell_rect * s_x)
                                y_min= int(y*self.params.dim_hog_cell_rect * s_y)
                                x_max= int((x*self.params.dim_hog_cell_rect + self.params.dim_window_w) * s_x)
                                y_max= int((y*self.params.dim_hog_cell_rect + self.params.dim_window_h) * s_y)
                                
                                image_detections.append([x_min, y_min, x_max, y_max])        
                                image_scores.append(score)
                    
                    if len(image_scores)!=0:
                        image_detections_all.append(image_detections)
                        image_scores_all.append(np.array(image_scores, dtype=np.float32) * 1.8)
                    
                
                if s == 0.4 or s == 0.6 or s == 0.8:
                # patrat 40 x 40 pixeli       
                    image_scores = []
                    image_detections = []
                
                
                    hog_descriptor_sq = hog(img_res, pixels_per_cell=(self.params.dim_hog_cell_sq, self.params.dim_hog_cell_sq),
                                cells_per_block=(2, 2), feature_vector=False)
                    
                    num_cols = img_res.shape[1]//self.params.dim_hog_cell_sq-1
                    num_rows = img_res.shape[0]//self.params.dim_hog_cell_sq-1
                    num_cel_template = self.params.dim_window_sq // self.params.dim_hog_cell_sq-1
                    
                    for y in range(0, num_rows-num_cel_template):
                        for x in range(0, num_cols-num_cel_template):
                            descr = hog_descriptor_sq[y:y+num_cel_template, x:x+num_cel_template].flatten()
                            score = np.dot(descr, w_sq)[0] + bias_sq
                            if score > self.params.threshold_sq:
                                s_x = img.shape[1] / width
                                s_y = img.shape[0] / height
                                
                                x_min= int(x*self.params.dim_hog_cell_sq * s_x)
                                y_min= int(y*self.params.dim_hog_cell_sq * s_y)
                                x_max= int((x*self.params.dim_hog_cell_sq + self.params.dim_window_sq) * s_x)
                                y_max= int((y*self.params.dim_hog_cell_sq + self.params.dim_window_sq) * s_y)
                                
                                image_detections.append([x_min, y_min, x_max, y_max])        
                                image_scores.append(score)
                    
                    if len(image_scores)!=0:
                        image_detections_all.append(image_detections)
                        image_scores_all.append(np.array(image_scores, dtype=np.float32) * 0.5)
                
                # dreptunghi 64 x 96 pixeli
                if  s!= 0.4  and s != 1.2:
                
                    image_scores = []
                    image_detections = []
                
                
                    hog_descriptor_inv = hog(img_res, pixels_per_cell=(self.params.dim_hog_cell_inv, self.params.dim_hog_cell_inv),
                                cells_per_block=(2, 2), feature_vector=False)
                    num_cols = img_res.shape[1]//self.params.dim_hog_cell_inv-1
                    num_rows = img_res.shape[0]//self.params.dim_hog_cell_inv-1
                    num_cel_template_w = self.params.dim_window_inv_w // self.params.dim_hog_cell_inv-1
                    num_cel_template_h = self.params.dim_window_inv_h // self.params.dim_hog_cell_inv-1
                    
                    
                    for y in range(0, num_rows-num_cel_template_h):
                        for x in range(0, num_cols-num_cel_template_w):
                            descr = hog_descriptor_inv[y:y+num_cel_template_h, x:x+num_cel_template_w].flatten()
                            score = np.dot(descr, w_inv)[0] + bias_inv
                            if score > self.params.threshold_inv:
                                s_x = img.shape[1] / width
                                s_y = img.shape[0] / height
                                
                                x_min= int(x*self.params.dim_hog_cell_inv * s_x)
                                y_min= int(y*self.params.dim_hog_cell_inv * s_y)
                                x_max= int((x*self.params.dim_hog_cell_inv + self.params.dim_window_inv_w) * s_x)
                                y_max= int((y*self.params.dim_hog_cell_inv + self.params.dim_window_inv_h) * s_y)
                                
                                image_detections.append([x_min, y_min, x_max, y_max])        
                                image_scores.append(score)
                    
                    if len(image_scores)!=0:
                        image_detections_all.append(image_detections)
                        image_scores_all.append(np.array(image_scores, dtype=np.float32) * 1.8)   
            
            
            if image_scores_all:
                image_detections, image_scores = self.non_maximal_suppression(np.concatenate(image_detections_all), np.concatenate(image_scores_all), img.shape) 
            
                
            if len(image_scores) !=0:
                if detections is None:
                    detections = image_detections
                else:
                    detections = np.concatenate((detections, image_detections))
                    
                scores = np.append(scores, image_scores)
                
                name = ntpath.basename(test_files[i])
                image_names = [name for _ in range(len(image_scores))]
                file_names = np.append(file_names, image_names)
            

            end_time = timeit.default_timer()
            print('Timpul de procesarea al imaginii de testare %d/%d este %f sec.'
                  % (i, num_test_images, end_time - start_time))

        return detections, scores, file_names

    
    
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

    def eval_detections(self, detections, scores, file_names):
        ground_truth_file = np.loadtxt(self.params.path_annotations, dtype='str')
        ground_truth_file_names = np.array(ground_truth_file[:, 0])
        ground_truth_detections = np.array(ground_truth_file[:, 1:5], np.int32)

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
            if max_overlap >= 0.3:   # nu trebuie sa modificam 0.3 !!!!!!!!!!!!!!!!!!!!!
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
        plt.title('Average precision %.3f' % average_precision)
        plt.savefig(os.path.join(self.params.dir_save_files, 'precizie_medie.png'))
        plt.show()
