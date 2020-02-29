import igraph as ig
from GMM import GaussianMixture
import numpy as np
import copy
import time

BLUE = [255,0,0]        # rectangle color
RED = [0,0,255]         # PR BG
GREEN = [0,255,0]       # PR FG
BLACK = [0,0,0]         # sure BG
WHITE = [255,255,255]   # sure FG

DRAW_BG = {'color' : BLACK, 'val' : 0}
DRAW_FG = {'color' : WHITE, 'val' : 1}
DRAW_PR_FG = {'color' : GREEN, 'val' : 3}
DRAW_PR_BG = {'color' : RED, 'val' : 2}
INF = 1000000

class GrabCut(object):
    def __init__(self, img, mask, rect = None, gmm_components = 5):
        self.img = np.asarray(img, dtype = np.float64)
        self.rows, self.cols, _ = img.shape
        self.gmm_components = gmm_components
        self.mask = mask
        self.gamma = 50

        if rect != None:
            self.mask[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] =  DRAW_PR_FG['val']

        self.indexes_classification()
        self.bgd_gmm = GaussianMixture(img[self.bgd_indexes], n_components = gmm_components)
        self.fgd_gmm = GaussianMixture(img[self.fgd_indexes], n_components = gmm_components)

        self.source = self.rows * self.cols
        self.target = self.rows * self.cols + 1
        self.calculate_beta_smoothness()
        #start = time.time()
        #self.build_graph_init()
        #end = time.time()
        #print("Build initial graph: ", start - end)
        self.run()

    def indexes_classification(self):
        self.bgd_indexes = np.where(np.logical_or(self.mask == DRAW_BG['val'], self.mask == DRAW_PR_BG['val']))
        self.fgd_indexes = np.where(np.logical_or(self.mask == DRAW_FG['val'], self.mask == DRAW_PR_FG['val']))

    def calculate_beta_smoothness(self):
        _left_diff = self.img[:, 1:] - self.img[:, :-1]
        _upleft_diff = self.img[1:, 1:] - self.img[:-1, :-1]
        _up_diff = self.img[1:, :] - self.img[:-1, :]
        _upright_diff = self.img[1:, :-1] - self.img[:-1, 1:]

        self.beta = np.sum(np.square(_left_diff)) + np.sum(np.square(_upleft_diff)) + \
            np.sum(np.square(_up_diff)) + \
            np.sum(np.square(_upright_diff))
        self.beta = 1 / (2 * self.beta / (
            # Each pixel has 4 neighbors (left, upleft, up, upright)
            4 * self.cols * self.rows
            # The 1st column doesn't have left, upleft and the last column doesn't have upright
            - 3 * self.cols
            - 3 * self.rows  # The first row doesn't have upleft, up and upright
            + 2))  # The first and last pixels in the 1st row are removed twice
        print(self.beta)

        self.left_V = self.gamma * np.exp(-self.beta * np.sum(np.square(_left_diff), axis=2))
        self.upleft_V = self.gamma / np.sqrt(2) * np.exp(-self.beta * np.sum(np.square(_upleft_diff), axis=2))
        self.up_V = self.gamma * np.exp(-self.beta * np.sum(np.square(_up_diff), axis=2))
        self.upright_V = self.gamma / np.sqrt(2) * np.exp(-self.beta * np.sum(np.square(_upright_diff), axis=2))

    def build_graph_init(self):
        edges = []
        capacity = []

        #Left edge adding
        for i in range(self.rows):
            for j in range(1, self.cols):
                edges.append(((i * self.cols + j), (i * self.cols + j - 1)))
                capacity.append(self.left_V[i,j-1])
                edges.append(((i * self.cols + j - 1), (i * self.cols + j)))
                capacity.append(self.left_V[i,j-1])

        #Left Up edge adding
        for i in range(1, self.rows):
            for j in range(1, self.cols):
                edges.append(((i * self.cols + j), ((i - 1) * self.cols + j - 1)))
                capacity.append(self.upleft_V[i - 1, j - 1])
                edges.append((((i - 1) * self.cols + j - 1), (i * self.cols + j)))
                capacity.append(self.upleft_V[i - 1, j - 1])

        #Up edge adding
        for i in range(1, self.rows):
            for j in range(self.cols):
                edges.append(((i * self.cols + j), ((i - 1) * self.cols + j)))
                capacity.append(self.up_V[i - 1, j])
                edges.append((((i - 1) * self.cols + j), (i * self.cols + j)))
                capacity.append(self.up_V[i - 1, j])

        #Right Up edge adding
        for i in range(1, self.rows):
            for j in range(self.cols - 1):
                edges.append(((i * self.cols + j), ((i - 1) * self.cols + j + 1)))
                capacity.append(self.upright_V[i - 1, j])
                edges.append((((i - 1) * self.cols + j + 1), (i * self.cols + j)))
                capacity.append(self.upright_V[i - 1, j])
        return edges, capacity

    def build_graph(self):
        bgd_indexes = np.where(self.mask == DRAW_BG['val'])
        fgd_indexes = np.where(self.mask == DRAW_FG['val'])
        pr_indexes = np.where(np.logical_or(self.mask == DRAW_PR_BG['val'], self.mask == DRAW_PR_FG['val']))
        edges, capacity = self.build_graph_init()

        bgd_indexes_zip = list(bgd_indexes[0] * self.cols + bgd_indexes[1])
        fgd_indexes_zip = list(fgd_indexes[0] * self.cols + fgd_indexes[1])
        pr_indexes_zip = list(pr_indexes[0] * self.cols + pr_indexes[1])

        #Edges from source
        #for v in fgd_indexes_zip:
        #    edges.append((self.source, v))
        #    capacity.append(INF)
        edges.extend(list(zip([self.source] * len(fgd_indexes_zip), fgd_indexes_zip)))
        capacity.extend([INF] * len(fgd_indexes_zip))

        #Edge to target
        #for v in bgd_indexes_zip:
        #    edges.append((v, self.target))
        #    capacity.append(INF)
        edges.extend(list(zip(bgd_indexes_zip, [self.target] * len(bgd_indexes_zip))))
        capacity.extend([INF] * len(bgd_indexes_zip))

        #Edge from source and to target
        prob_bgd = -np.log(self.bgd_gmm.calculate_probability(self.img[pr_indexes]))
        prob_fgd = -np.log(self.fgd_gmm.calculate_probability(self.img[pr_indexes]))
        #for index in range(len(pr_indexes_zip)):
        #    v = pr_indexes_zip[index]
        #    edges.append((self.source, v))
        #    capacity.append(prob_bgd[index])
        #    edges.append((v, self.target))
        #    capacity.append(prob_fgd[index])
        edges.extend(list(zip([self.source] * len(pr_indexes_zip), pr_indexes_zip)))
        capacity.extend(list(prob_bgd))
        edges.extend(list(zip(pr_indexes_zip, [self.target] * len(pr_indexes_zip))))
        capacity.extend(list(prob_fgd))
        return edges, capacity

    def min_cut_segmentation(self):
        edges, capacity = self.build_graph()
        graph = ig.Graph(self.rows * self.cols + 2)
        graph.add_edges(edges)
        mincut = graph.st_mincut(self.source, self.target, capacity)
        pr_indexes = np.where(np.logical_or(
            self.mask == DRAW_PR_BG['val'], self.mask == DRAW_PR_FG['val']))
        img_indexes = np.arange(self.rows * self.cols,
                                dtype=np.uint32).reshape(self.rows, self.cols)
        self.mask[pr_indexes] = np.where(np.isin(img_indexes[pr_indexes], mincut.partition[0]),
                                         DRAW_PR_FG['val'], DRAW_PR_BG['val'])
        self.indexes_classification()

    def run(self, n_iters = 1, skip_learn_GMMs = False):
        for _ in range(n_iters):
            if not skip_learn_GMMs:
                bgd_labels = self.bgd_gmm.components_registration(self.img[self.bgd_indexes])
                self.bgd_gmm.fit(self.img[self.bgd_indexes], bgd_labels)
                fgd_labels = self.fgd_gmm.components_registration(self.img[self.fgd_indexes])
                self.fgd_gmm.fit(self.img[self.fgd_indexes], fgd_labels)
            self.min_cut_segmentation()
            skip_learn_GMMs = False
