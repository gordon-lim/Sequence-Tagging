import localized_narratives
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pickle as pkl
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from PIL import Image
import itertools
import string


# https://github.com/google/localized-narratives/blob/master/demo.py
local_dir = os.getcwd()

data_loader = localized_narratives.DataLoader(local_dir)

data_loader.download_annotations('ade20k_val')

# Get object name lookup table
with open('/Volumes/gordonssd/ADE20K_2021_17_01/index_ade20k.pkl', 'rb') as f:
    index_ade20k = pkl.load(f)
# Replace "-" with "other"
index_ade20k['objectnames'][0] = "other"


def get_documents(num_narratives):
    loc_narrs = data_loader.load_annotations(
        'ade20k_val', num_narratives)  # Change number of narratives used

    raw_word_documents = []
    raw_labels_documents = []

    for loc_narr in loc_narrs:
        raw_labels_doc = ''
        raw_word_documents.append(clean_text(loc_narr.caption))
        image_id = loc_narr.image_id
        path_to_image = f'/Volumes/gordonssd/ADE20K_2021_17_01/images/ADE/validation/*/*/{image_id}.jpg'
        path_to_image_seg = f'/Volumes/gordonssd/ADE20K_2021_17_01/images/ADE/validation/*/*/{image_id}_seg.png'
        path_to_image_json = f'/Volumes/gordonssd/ADE20K_2021_17_01/images/ADE/validation/*/*/{image_id}.json'
        # Get image
        for filename in glob.glob(path_to_image):
            path_to_image = filename
        image = mpimg.imread(path_to_image)
        # Get image segmentation
        for filename in glob.glob(path_to_image_seg):
            path_to_image_seg = filename
        image_seg = mpimg.imread(path_to_image_seg)
        # Get image height, width
        image_height = image.shape[0]
        image_width = image.shape[1]
        # Get Object Labels
        with Image.open(path_to_image_seg) as io:
            seg = np.array(io)
        R = seg[:, :, 0]
        G = seg[:, :, 1]
        B = seg[:, :, 2]
        ObjectClassMasks = (R/10).astype(np.int32)*256+(G.astype(np.int32))
        # Concatenate all trace segments
        traces = list(itertools.chain.from_iterable(loc_narr.traces))
        # Align traces to word #
        # Get list of word_trace_align objects
        save_word = ''
        for word in loc_narr.timed_caption:
            w = save_word + word['utterance']
            w = clean_text(w)
            save_word = ''
            # Check if start_time match end_time
            if (word['start_time'] == word['end_time']):
                # Make phrase with next word
                save_word = word['utterance'] + ' '
                continue
            # Convert (start_time, end_time) to trace coordinates that fall within the time window
            start_time = word['start_time']
            end_time = word['end_time']
            # Filter trace_seg for items with t value within start_time:end_time
            points = list(filter(lambda coord: start_time <=
                          coord['t'] <= end_time, traces))
            points = np.array([[point['x'], point['y']] for point in points])
            try:
                hull = ConvexHull(points)
            except:
                pass
            # hull_to_name
            # make a polygon from the hull verticies
            points = hull.points
            points = np.array([[point[0]*image_width, point[1]*image_height]
                               for point in points])
            tupVerts = [(points[vtx, 0], points[vtx, 1])
                        for vtx in hull.vertices]
            p = Path(tupVerts)
            # make a canvas of coordinates corresponding to the image
            x, y = np.meshgrid(np.arange(image_width), np.arange(image_height))
            x, y = x.flatten(), y.flatten()
            points = np.vstack((x, y)).T
            # get points in image within hull
            grid = p.contains_points(points)
            # now you have a mask with points inside a polygon
            mask = grid.reshape(image_height, image_width)
            # get labels at those points
            labels_within_hull = mask*ObjectClassMasks
            labels = labels_within_hull[np.nonzero(labels_within_hull)]
            # get count for unique labels
            values, counts = np.unique(labels, return_counts=True)
            # if empty set label to 0
            # Edge case: some utterances the traces entirely out of the image
            if len(counts) == 0:
                tag = index_ade20k['objectnames'][0]
            else:
                # get most frequent label within hull
                ind = np.argmax(counts)
                label = values[ind]
                # object name lookup with label
                tag = index_ade20k['objectnames'][label - 1]
            # one tag may have multiple words eg. 	person;individual;someone;somebody;mortal;soul
            # taking only first
            tag = tag.split()[0]
            tag = clean_text(tag)
            # Add labels to labels document
            for i in range(len(w.split())):
                raw_labels_doc += tag + ' '
        raw_labels_documents.append(raw_labels_doc.strip())
    return raw_word_documents, raw_labels_documents

##################
####HELPERS#######
##################


def clean_text(text):
    return text.translate(str.maketrans('', '', string.punctuation)).lower()

# for printing with colors


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
