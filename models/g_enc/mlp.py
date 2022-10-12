import pdb
import numpy as np
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from data_eval import InstrumentDataset, save_audio
from data import RenderedInstrumentDataset, RenderedMultiInstrumentDataset
from byol_a2.common import (np, Path, torch,
     get_logger, load_yaml_config, seed_everything, get_timestamp, hash_text)
import csv
import copy
import scipy
from numpy import zeros, ones, array, arange, argsort, cumsum, column_stack, hstack, vstack, isscalar, abs, concatenate, exp, log, inf, infty
from numpy.linalg import solve
from collections import namedtuple
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.metrics.pairwise import cosine_similarity
import os

Box = namedtuple("Box", "left right top bottom")

class MLP(nn.Module):
    def __init__(self, multi = False):
        super().__init__()
        self.layer1 = nn.Linear(1024,2048)
        self.layer2 = nn.Linear(2048,1024)
        self.layer3 = nn.Linear(1024,953)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.multi = multi

    def forward(self,x,return_embedding=False):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.layer2(x)
        x = self.relu(out)
        x = self.dropout(x)
        x = self.layer3(x)
        if return_embedding:
            return x, out
        elif self.multi:
            return out
        else:
            return x

def train(dataloader, model, loss_fn, optimizer,device, multi=False):
    if multi:
        for batch, (X,y) in enumerate(dataloader):
            print(batch, "excuting")
            pred = model(X.to(device),False)
            optimizer.zero_grad()
            loss = loss_fn(pred, y.to(device))
            loss.backward()
            optimizer.step()
        return loss
    else:
        for batch, (X,y) in enumerate(dataloader):
            pred = model(X, False)
            optimizer.zero_grad()
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            pred,emb = model(X,True)
            byol_EER = compute_eer_det(X.cpu(), y.cpu())
            EER = compute_eer_det(emb.cpu(), y.cpu())
        return loss, EER, byol_EER

def test(dataloader, model, loss_fn, multi=False):
    encoder = MLP(multi=False)
    encoder.load_state_dict(torch.load("/home/slslslrhfem/byol-acreationview/v2/checkpoints/MLPmodel.pt"))
    if multi:
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        with torch.no_grad():
            for X,y in dataloader:
                pred = model(X, False)
                loss += loss_fn(pred,y).item()
        return loss / num_batches
    else:
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        loss, correct = 0, 0
        with torch.no_grad():
            for X,y in dataloader:
                pred = model(X,False)
                loss += loss_fn(pred,y).item()
                correct += (pred.argmax(1)==y).type(torch.float).sum().item()
            pred,emb = model(X,True)
            valid_EER = compute_eer_det(emb.cpu(), y.cpu())
        loss /= num_batches
        correct /= size

        #print(f'Test Accuracy: {(100*correct):>0.1f}%     Loss: {loss:>8f} \n')

        return 100*correct, loss, valid_EER

def get_embedding(X,y, model):
    pred, embedding = model(X, True)
    EER = compute_eer_det(embedding.cpu(), y.cpu())
    return embedding, EER



def compute_eer_det(feat, label, enroll_num=5, tar_num=20, nontar_num=20, save_dir=None): 
    #feat = encoded vector = model.forward(data_A)[0],   label = label_A (데이터로드후 튜플 중 [1]을 인덱싱해서 가져온 정답 레이블)
    # feat: n*d
    # label: n
    np.random.seed(1024)
    cls_label = np.unique(label)
    tar_scores = []
    nontar_scores = []

    for i in cls_label:
        # __import__('ipdb').set_trace()
        cls_index = np.where(label==i)[0]

        enroll_index = np.random.choice(cls_index, enroll_num)

        tar_index = np.setdiff1d(cls_index, enroll_index)

        nontar_index = np.setdiff1d(np.arange(label.shape[0]), cls_index)

        for item in tar_index:
            # __import__('ipdb').set_trace()
            tar_index_sub = np.random.choice(tar_index, tar_num-1)
            tar_index_sub = np.concatenate([np.array([item]), tar_index_sub])
            nontar_index_sub = np.random.choice(nontar_index, nontar_num)
            # tar_index = np.random.choice(nontar_index, tar_num)
            # nontar_index = np.random.choice(nontar_index, nontar_num)

            # extrace feature for enrollment, target, and non-target
            enroll_feat = feat[enroll_index].mean(axis=0, keepdims=True)
            tar_feat = feat[tar_index_sub]
            nontar_feat = feat[nontar_index_sub]

            # compute cosine similarity score between enrollment and tar/non-tar
            tar_score = cosine_similarity(enroll_feat, tar_feat)
            nontar_score = cosine_similarity(enroll_feat, nontar_feat)
            tar_scores.append(tar_score.squeeze())
            nontar_scores.append(nontar_score.squeeze())

    tar_scores = np.hstack(tar_scores)
    nontar_scores = np.hstack(nontar_scores)
    if save_dir is not None:
        np.save(os.path.join(save_dir, 'tar_scores.npy'), tar_scores)
        np.save(os.path.join(save_dir, 'nontar_scores.npy'), nontar_scores)
    _, _, eer, _ = rocchdet(tar_scores, nontar_scores)
    return eer

def rocchdet(tar, non,
             dcfweights=array([]),
             pfa_min=5e-4,
             pfa_max=0.5,
             pmiss_min=5e-4,
             pmiss_max=0.5,
             dps=100,
             normalize=False):
    """ROCCHDET: Computes ROC Convex Hull and then maps that to the DET axes.
    The DET-curve is infinite, non-trivial limits (away from 0 and 1)
    are mandatory.

    :param tar: vector of target scores
    :param non: vector of non-target scores
    :param dcfweights: 2-vector, such that: DCF = [pmiss,pfa]*dcfweights(:)  (Optional, provide only if mindcf is
    desired, otherwise omit or use []
    :param pfa_min: limit of DET-curve rectangle. Default is 0.0005
    :param pfa_max: limit of DET-curve rectangle. Default is 0.5
    :param pmiss_min: limit of DET-curve rectangle. Default is 0.0005
    :param pmiss_max: limits of DET-curve rectangle.  Default is 0.5
    :param dps: number of returned (x,y) dots (arranged in a curve) in DET space, for every straight line-segment
    (edge) of the ROC Convex Hull. Default is 100.
    :param normalize: normalize the curve

    :return: probit(Pfa)
    :return: probit(Pmiss)
    :return: ROCCH EER = max_p mindcf(dcfweights=[p,1-p]), which is also equal to the intersection of the ROCCH
    with the line pfa = pmiss.
    :return: the mindcf: Identical to result using traditional ROC, but computed by mimimizing over the ROCCH
    vertices, rather than over all the ROC points.
    """
    assert ((pfa_min > 0) & (pfa_max < 1) & (pmiss_min > 0) & (pmiss_max < 1)), 'limits must be strictly inside (0,1)'
    assert ((pfa_min < pfa_max) & (pmiss_min < pmiss_max)), 'pfa and pmiss min and max values are not consistent'

    pmiss, pfa = rocch(tar, non)
    mindcf = 0.0

    if dcfweights.shape == (2,):
        dcf = dcfweights @ vstack((pmiss, pfa))
        mindcf = dcf.min()
        if normalize:
            mindcf = mindcf / min(dcfweights)

    # pfa is decreasing
    # pmiss is increasing
    box = Box(left=pfa_min, right=pfa_max, top=pmiss_max, bottom=pmiss_min)
    x = []
    y = []
    eer = 0
    for i in range(pfa.shape[0] - 1):
        xx = pfa[i:i + 2]
        yy = pmiss[i:i + 2]
        xdots, ydots, eerseg = plotseg(xx, yy, box, dps)
        x = x + xdots.tolist()
        y = y + ydots.tolist()
        eer = max(eer, eerseg)
    return array(x), array(y), eer, mindcf

def rocch(tar_scores, nontar_scores):
    pmiss, pfa, _, _ = rocch_pava(tar_scores, nontar_scores)
    return pmiss, pfa

def plotseg(xx, yy, box, dps):
    """Prepare the plotting of a curve.
    :param xx:
    :param yy:
    :param box:
    :param dps:
    """
    assert ((xx[1] <= xx[0]) & (yy[0] <= yy[1])), 'xx and yy should be sorted'

    XY = column_stack((xx, yy))
    dd = array([1, -1]) @  XY
    if abs(dd).min() == 0:
        eer = 0
    else:
        # find line coefficients seg s.t. seg'[xx(i);yy(i)] = 1,
        # when xx(i),yy(i) is on the line.
        seg = solve(XY, array([[1], [1]]))
        # candidate for EER, eer is highest candidate
        eer = 1.0 / seg.sum()

    # segment completely outside of box
    if (xx[0] < box.left) | (xx[1] > box.right) | (yy[1] < box.bottom) | (yy[0] > box.top):
        x = array([])
        y = array([])
    else:
        if xx[1] < box.left:
            xx[1] = box.left
            yy[1] = (1 - seg[0] * box.left) / seg[1]

        if xx[0] > box.right:
            xx[0] = box.right
            yy[0] = (1 - seg[0] * box.right) / seg[1]

        if yy[0] < box.bottom:
            yy[0] = box.bottom
            xx[0] = (1 - seg[1] * box.bottom) / seg[0]

        if yy[1] > box.top:
            yy[1] = box.top
            xx[1] = (1 - seg[1] * box.top) / seg[0]

        dx = xx[1] - xx[0]
        xdots = xx[0] + dx * arange(dps + 1) / dps
        ydots = (1 - seg[0] * xdots) / seg[1]
        x = probit(xdots)
        y = probit(ydots)

    return x, y, eer

def rocch_pava(tar_scores, nontar_scores, laplace=False):
    """ROCCH: ROC Convex Hull.
    Note: pmiss and pfa contain the coordinates of the vertices of the
    ROC Convex Hull.

    :param tar_scores: vector of target scores
    :param nontar_scores: vector of non-target scores

    :return: a tupple of two vectors: Pmiss, Pfa
    """
    Nt = tar_scores.shape[0]
    Nn = nontar_scores.shape[0]
    N = Nt + Nn
    scores = concatenate((tar_scores, nontar_scores))
    # Pideal is the ideal, but non-monotonic posterior
    Pideal = concatenate((ones(Nt), zeros(Nn)))

    # It is important here that scores that are the same
    # (i.e. already in order) should NOT be swapped.rb
    perturb = argsort(scores, kind='mergesort')
    #
    Pideal = Pideal[perturb]

    if laplace:
       Pideal = hstack([1,0,Pideal,1,0])

    Popt, width, foo = pavx(Pideal)

    if laplace:
      Popt = Popt[2:len(Popt)-2]

    nbins = width.shape[0]
    pmiss = zeros(nbins + 1)
    pfa = zeros(nbins + 1)

    # threshold leftmost: accept everything, miss nothing
    left = 0  # 0 scores to left of threshold
    fa = Nn
    miss = 0

    for i in range(nbins):
        pmiss[i] = miss / Nt
        pfa[i] = fa / Nn
        left = int(left + width[i])
        miss = Pideal[:left].sum()
        fa = N - left - Pideal[left:].sum()

    pmiss[nbins] = miss / Nt
    pfa[nbins] = fa / Nn

    return pmiss, pfa, Popt, perturb

def diff(list1, list2):
    c = [item for item in list1 if item not in list2]
    c.sort()
    return c

def ismember(list1, list2):
    c = [item in list2 for item in list1]
    return c

def probit(p):
    """Map from [0,1] to [-inf,inf] as used to make DET out of a ROC

    :param p: the value to map

    :return: probit(input)
    """
    y = np.sqrt(2) * scipy.special.erfinv(2 * p - 1)
    return y

def logit(p):
    """logit function.
    This is a one-to-one mapping from probability to log-odds.
    i.e. it maps the interval (0,1) to the real line.
    The inverse function is given by SIGMOID.

    log_odds = logit(p) = log(p/(1-p))

    :param p: the input value

    :return: logit(input)
    """
    p = np.array(p)
    lp = np.zeros(p.shape)
    f0 = p == 0
    f1 = p == 1
    f = (p > 0) & (p < 1)

    if lp.shape == ():
        if f:
            lp = np.log(p / (1 - p))
        elif f0:
            lp = -np.inf
        elif f1:
            lp = np.inf
    else:
        lp[f] = np.log(p[f] / (1 - p[f]))
        lp[f0] = -np.inf
        lp[f1] = np.inf
    return lp

def sigmoid(log_odds):
    """SIGMOID: Inverse of the logit function.
    This is a one-to-one mapping from log odds to probability.
    i.e. it maps the real line to the interval (0,1).

    p = sigmoid(log_odds)

    :param log_odds: the input value

    :return: sigmoid(input)
    """
    p = 1 / (1 + np.exp(-log_odds))
    return p

def pavx(y):
    """PAV: Pool Adjacent Violators algorithm.
    Non-paramtetric optimization subject to monotonicity.

    ghat = pav(y)
    fits a vector ghat with nondecreasing components to the
    data vector y such that sum((y - ghat).^2) is minimal.
    (Pool-adjacent-violators algorithm).

    optional outputs:
            width: width of pav bins, from left to right
                    (the number of bins is data dependent)
            height: corresponding heights of bins (in increasing order)

    Author: This code is a simplified version of the 'IsoMeans.m' code
    made available by Lutz Duembgen at:
    http://www.imsv.unibe.ch/~duembgen/software

    :param y: input value
    """
    assert y.ndim == 1, 'Argument should be a 1-D array'
    assert y.shape[0] > 0, 'Input array is empty'
    n = y.shape[0]

    index = np.zeros(n,dtype=int)
    length = np.zeros(n,dtype=int)

    ghat = np.zeros(n)

    ci = 0
    index[ci] = 1
    length[ci] = 1
    ghat[ci] = y[0]

    for j in range(1, n):
        ci += 1
        index[ci] = j+1
        length[ci] = 1
        ghat[ci] = y[j]
        while (ci >= 1) & (ghat[np.max(ci - 1, 0)] >= ghat[ci]):
            nw = length[ci - 1] + length[ci]
            ghat[ci - 1] = ghat[ci - 1] + (length[ci] / nw) * (ghat[ci] - ghat[ci - 1])
            length[ci - 1] = nw
            ci -= 1

    height = copy.deepcopy(ghat[:ci + 1])
    width = copy.deepcopy(length[:ci + 1])

    while n >= 0:
        for j in range(index[ci], n+1):
            ghat[j-1] = ghat[ci]
        n = index[ci] - 1
        ci -= 1

    return ghat, width, height

def pavx_mapping(tar, non, monotonicity_epsilon=1e-6, large_value=1e6):
    # TODO unit test
    
    def map(scores, score_bounds, llr_bounds):
        p = (scores[:, None] >= score_bounds).sum(axis=1) - 1
        p2 = p + 1
        idx = p2 < len(score_bounds)  # failsafe for LLRs > 1e6; should not happen at all...
        
        v1 = llr_bounds[p]

        x1 = score_bounds[p[idx]]
        x2 = score_bounds[p2[idx]]
        v2 = llr_bounds[p2[idx]]

        v1[idx] += (scores[idx] - x1) / (x2 - x1) * (v2 - v1[idx])

        return v1


    scores = np.concatenate([[-large_value], non, tar, [large_value]])
    Pideal = np.concatenate([[1], np.zeros(len(non)), np.ones(len(tar)), [0]])
    perturb = np.argsort(scores, kind='mergesort')
    scores = scores[perturb]
    Pideal = Pideal[perturb]
    
    Popt, width, foo = pavx(Pideal)
    data_prior = (len(tar) + 1)/len(Pideal)
    llrs = logit(Popt) - logit(data_prior)
    
    # make bounds
    bnd_len = 2 * len(width)
    c = np.cumsum(width - 1)
    bnd_ndx = np.zeros(bnd_len)
    bnd_ndx[::2] = np.concatenate([[0], c[:-1]+1])
    bnd_ndx[1::2] = c + 1
    score_bounds = scores[bnd_ndx.astype(int)]
    llr_bounds = llrs[bnd_ndx.astype(int)]
    llr_bounds[::2] = llr_bounds[::2] - monotonicity_epsilon
    llr_bounds[1::2] = llr_bounds[1::2] + monotonicity_epsilon

    return lambda s: map(s, score_bounds=score_bounds, llr_bounds=llr_bounds)

def optimal_llr_from_Popt(Popt, perturb, Ntar, Nnon, monotonicity_epsilon=1e-6):
    posterior_log_odds = logit(Popt)
    log_prior_odds = np.log(Ntar/Nnon)
    llrs = posterior_log_odds - log_prior_odds
    N = Ntar + Nnon
    llrs = llrs + np.arange(N) * monotonicity_epsilon/N # preserve monotonicity

    idx_reverse = np.zeros(N, dtype=int)
    idx_reverse[perturb] = np.arange(N)
    llrs_reverse = llrs[idx_reverse]
    tar_llrs = llrs_reverse[:Ntar]
    nontar_llrs = llrs_reverse[Ntar:]

    return tar_llrs, nontar_llrs

def optimal_llr(tar, non, laplace=False, monotonicity_epsilon=1e-6):
    # flag Laplace: avoids infinite LLR magnitudes;
    # also, this stops DET cureves from 'curling' to the axes on sparse data (DETs stay in more populated regions)
    scores = np.concatenate([tar, non])
    Pideal = np.concatenate([np.ones(len(tar)), np.zeros(len(non))])

    perturb = np.argsort(scores, kind='mergesort')
    Pideal = Pideal[perturb]

    if laplace:
       Pideal = np.hstack([1,0,Pideal,1,0])

    Popt, width, foo = pavx(Pideal)

    if laplace:
      Popt = Popt[2:len(Popt)-2]

    tar_llrs, nontar_llrs = optimal_llr_from_Popt(Popt=Popt, perturb=perturb, Ntar=len(tar), Nnon=len(non), monotonicity_epsilon=monotonicity_epsilon)
    return tar_llrs, nontar_llrs

def neglogsigmoid(log_odds):
    neg_log_p = -log_odds
    e = np.exp(-log_odds)
    f = np.argwhere(e < e+1)[0]
    neg_log_p[f] = np.log(1+e[f])
    return neg_log_p
