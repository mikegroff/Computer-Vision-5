import numpy as np
import cyvlfeat as vlfeat
from utils import *
import os.path as osp
from glob import glob
from random import shuffle
from IPython.core.debugger import set_trace
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import cv2


def get_positive_features(train_path_pos, feature_params):
    """
    This function should return all positive training examples (faces) from
    36x36 images in 'train_path_pos'. Each face should be converted into a
    HoG template according to 'feature_params'.

    Useful functions:
    -   vlfeat.hog.hog(im, cell_size): computes HoG features

    Args:
    -   train_path_pos: (string) This directory contains 36x36 face images
    -   feature_params: dictionary of HoG feature computation parameters.
        You can include various parameters in it. Two defaults are:
            -   template_size: (default 36) The number of pixels spanned by
            each train/test template.
            -   hog_cell_size: (default 6) The number of pixels in each HoG
            cell. template size should be evenly divisible by hog_cell_size.
            Smaller HoG cell sizes tend to work better, but they make things
            slower because the feature dimensionality increases and more
            importantly the step size of the classifier decreases at test time
            (although you don't have to make the detector step size equal a
            single HoG cell).

    Returns:
    -   feats: N x D matrix where N is the number of faces and D is the template
            dimensionality, which would be (feature_params['template_size'] /
            feature_params['hog_cell_size'])^2 * 31 if you're using the default
            hog parameters.
    """
    # params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)
    n_cell = np.ceil(win_size/cell_size).astype('int')

    positive_files = glob(osp.join(train_path_pos, '*.jpg'))

    feats = np.random.rand(1, n_cell*n_cell*31)
    for im in positive_files:
        im = load_image_gray(im)
        feat = vlfeat.hog.hog(im, cell_size)
        feat = np.reshape(feat, (1,n_cell*n_cell*31))
        feats = np.append(feats,feat,axis = 0)
    feats = feats[1:]

    return feats

def get_random_negative_features(non_face_scn_path, feature_params, num_samples):
    """
    This function should return negative training examples (non-faces) from any
    images in 'non_face_scn_path'. Images should be loaded in grayscale because
    the positive training data is only available in grayscale (use
    load_image_gray()).

    Useful functions:
    -   vlfeat.hog.hog(im, cell_size): computes HoG features

    Args:
    -   non_face_scn_path: string. This directory contains many images which
            have no faces in them.
    -   feature_params: dictionary of HoG feature computation parameters. See
            the documentation for get_positive_features() for more information.
    -   num_samples: number of negatives to be mined. It is not important for
            the function to find exactly 'num_samples' non-face features. For
            example, you might try to sample some number from each image, but
            some images might be too small to find enough.

    Returns:
    -   N x D matrix where N is the number of non-faces and D is the feature
            dimensionality, which would be (feature_params['template_size'] /
            feature_params['hog_cell_size'])^2 * 31 if you're using the default
            hog parameters.
    """
    # params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)

    negative_files = glob(osp.join(non_face_scn_path, '*.jpg'))

    pi = int(num_samples/len(negative_files))+1

    n_cell = np.ceil(win_size/cell_size).astype('int')
    feats = np.random.rand(1, n_cell*n_cell*31)


    for im in negative_files:
        im = load_image_gray(im)
        im = vlfeat.hog.hog(im, cell_size)
        x,y,h = np.shape(im)
        x-=cell_size
        y-=cell_size
        for k in range(0,pi):
            i = np.random.randint(x,size = 1)
            j = np.random.randint(y,size = 1)
            i = i[0]
            j = j[0]
            feat = im[i:i+cell_size,j:j+cell_size,:]
            feat = np.reshape(feat, (1,n_cell*n_cell*31))
            feats = np.append(feats,feat,axis = 0)


    feats = feats[1:]
    return feats

def train_classifier(features_pos, features_neg, C):
    """
    This function trains a linear SVM classifier on the positive and negative
    features obtained from the previous steps. We fit a model to the features
    and return the svm object.

    Args:
    -   features_pos: N X D array. This contains an array of positive features
            extracted from get_positive_feats().
    -   features_neg: M X D array. This contains an array of negative features
            extracted from get_negative_feats().

    Returns:
    -   svm: LinearSVC object. This returns a SVM classifier object trained
            on the positive and negative features.
    """

    n,d = np.shape(features_pos)
    m,d = np.shape(features_neg)

    svm = LinearSVC(random_state=0, tol=1e-4, loss='hinge',max_iter = 10000, C=C)
    feats = np.append(features_pos,features_neg,axis = 0)
    labels = np.append(np.ones(n), np.zeros(m),axis = 0)
    svm.fit(feats, labels)


    return svm

def mine_hard_negs(non_face_scn_path, svm, feature_params):
    """
    This function is pretty similar to get_random_negative_features(). The only
    difference is that instead of returning all the extracted features, you only
    return the features with false-positive prediction.

    Useful functions:
    -   vlfeat.hog.hog(im, cell_size): computes HoG features
    -   svm.predict(feat): predict features

    Args:
    -   non_face_scn_path: string. This directory contains many images which
            have no faces in them.
    -   feature_params: dictionary of HoG feature computation parameters. See
            the documentation for get_positive_features() for more information.
    -   svm: LinearSVC object

    Returns:
    -   N x D matrix where N is the number of non-faces which are
            false-positive and D is the feature dimensionality.
    """
    num_samples = 100000

    # params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)

    negative_files = glob(osp.join(non_face_scn_path, '*.jpg'))
    pi = int(num_samples/len(negative_files))+1
    n_cell = np.ceil(win_size/cell_size).astype('int')
    feats = np.random.rand(1, n_cell*n_cell*31)
    for im in negative_files:
        im = load_image_gray(im)
        im = vlfeat.hog.hog(im, cell_size)
        x,y,h = np.shape(im)
        x-=cell_size
        y-=cell_size
        for k in range(0,pi):
            i = np.random.randint(x,size = 1)
            j = np.random.randint(y,size = 1)
            i = i[0]
            j = j[0]
            feat = im[i:i+cell_size,j:j+cell_size,:]
            feat = np.reshape(feat, (1,n_cell*n_cell*31))
            if svm.predict(feat) == 1.0:
                feats = np.append(feats,feat,axis = 0)


    feats = feats[1:]

    return feats

def run_detector(test_scn_path, svm, feature_params, verbose=False):
    """
    This function returns detections on all of the images in a given path. You
    will want to use non-maximum suppression on your detections or your
    performance will be poor (the evaluation counts a duplicate detection as
    wrong). The non-maximum suppression is done on a per-image basis. The
    starter code includes a call to a provided non-max suppression function.

    The placeholder version of this code will return random bounding boxes in
    each test image. It will even do non-maximum suppression on the random
    bounding boxes to give you an example of how to call the function.

    Your actual code should convert each test image to HoG feature space with
    a _single_ call to vlfeat.hog.hog() for each scale. Then step over the HoG
    cells, taking groups of cells that are the same size as your learned
    template, and classifying them. If the classification is above some
    confidence, keep the detection and then pass all the detections for an
    image to non-maximum suppression. For your initial debugging, you can
    operate only at a single scale and you can skip calling non-maximum
    suppression. Err on the side of having a low confidence threshold (even
    less than zero) to achieve high enough recall.

    Args:
    -   test_scn_path: (string) This directory contains images which may or
            may not have faces in them. This function should work for the
            MIT+CMU test set but also for any other images (e.g. class photos).
    -   svm: A trained sklearn.svm.LinearSVC object
    -   feature_params: dictionary of HoG feature computation parameters.
        You can include various parameters in it. Two defaults are:
            -   template_size: (default 36) The number of pixels spanned by
            each train/test template.
            -   hog_cell_size: (default 6) The number of pixels in each HoG
            cell. template size should be evenly divisible by hog_cell_size.
            Smaller HoG cell sizes tend to work better, but they make things
            slowerbecause the feature dimensionality increases and more
            importantly the step size of the classifier decreases at test time.
    -   verbose: prints out debug information if True

    Returns:
    -   bboxes: N x 4 numpy array. N is the number of detections.
            bboxes(i,:) is [x_min, y_min, x_max, y_max] for detection i.
    -   confidences: (N, ) size numpy array. confidences(i) is the real-valued
            confidence of detection i.
    -   image_ids: List with N elements. image_ids[i] is the image file name
            for detection i. (not the full path, just 'albert.jpg')
    """
    im_filenames = sorted(glob(osp.join(test_scn_path, '*.jpg')))
    bboxes = np.empty((0, 4))
    confidences = np.empty(0)
    image_ids = []

    # number of top detections to feed to NMS
    topk = 250

    # params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)
    scale_factor = feature_params.get('scale_factor', 0.65)
    template_size = int(win_size / cell_size)

    for idx, im_filename in enumerate(im_filenames):
        print('Detecting faces in {:s}'.format(im_filename))
        img = load_image_gray(im_filename)
        im_id = osp.split(im_filename)[-1]
        im_shape = img.shape
        l,p = np.shape(img)
        l = (win_size/l)
        p = (win_size/p)
        sf = [l*p,0.01,0.15,0.25,0.35,0.5,0.65,0.75,0.9]
        cf = -0.7
        cur_x_min = []
        cur_x_max = []
        cur_y_min = []
        cur_y_max = []
        cur_confidences = []
        c_size = 1
        for s in sf:
            im = cv2.resize(img,None,fx=s,fy=s)
            # create scale space HOG pyramid and return scores for prediction
            im = vlfeat.hog.hog(im, cell_size)
            x,y,h = np.shape(im)
            m = np.floor(x/(c_size)).astype(int)
            n = np.floor(y/(c_size)).astype(int)
            m -= int((cell_size-x+m)/c_size)
            n -= int((cell_size-y+n)/c_size)
            for i in range(0,m):
                for j in range(0,n):
                    a,b,c,d = i*c_size,i*c_size+cell_size,j*c_size,j*c_size+cell_size
                    feat = im[a:b,c:d,:]
                    #print(x,y,m,n,a,b,c,d)
                    feat = np.reshape(feat, (1,template_size*template_size*31))
                    con = np.dot(svm.coef_,feat.T)[0,0]+ svm.intercept_[0]
                    if con > cf:
                        cur_y_min.append((a*cell_size)/s)
                        cur_y_max.append((b*cell_size)/s)
                        cur_x_min.append((c*cell_size)/s)
                        cur_x_max.append((d*cell_size)/s)
                        cur_confidences.append(con)

        cur_x_min,cur_x_max,cur_y_min, cur_y_max = np.asarray([cur_x_min]), np.asarray([cur_y_min]), np.asarray([cur_x_max]), np.asarray([cur_y_max])
        cur_bboxes = np.hstack([cur_x_min.T,cur_x_max.T,cur_y_min.T, cur_y_max.T])
        cur_confidences = np.asarray(cur_confidences)




        ### non-maximum suppression ###
        # non_max_supr_bbox() can actually get somewhat slow with thousands of
        # initial detections. You could pre-filter the detections by confidence,
        # e.g. a detection with confidence -1.1 will probably never be
        # meaningful. You probably _don't_ want to threshold at 0.0, though. You
        # can get higher recall with a lower threshold. You should not modify
        # anything in non_max_supr_bbox(). If you want to try your own NMS methods,
        # please create another function.

        idsort = np.argsort(-cur_confidences)[:topk]
        cur_bboxes = cur_bboxes[idsort]
        cur_confidences = cur_confidences[idsort]

        is_valid_bbox = non_max_suppression_bbox(cur_bboxes, cur_confidences,
            im_shape, verbose=verbose)

        print('NMS done, {:d} detections passed'.format(sum(is_valid_bbox)))

        if(sum(is_valid_bbox) == 0):
            continue
        cur_bboxes = cur_bboxes[is_valid_bbox]
        cur_confidences = cur_confidences[is_valid_bbox]

        bboxes = np.vstack((bboxes, cur_bboxes))
        confidences = np.hstack((confidences, cur_confidences))
        image_ids.extend([im_id] * len(cur_confidences))

    return bboxes, confidences, image_ids
