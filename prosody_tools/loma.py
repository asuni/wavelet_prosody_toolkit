import numpy as np
from operator import itemgetter

# Logging
import logging
logger = logging.getLogger(__name__)

def save_analyses(fname, labels, prominences, boundaries, frame_rate=200, with_header=False):
    """Save analysis into a csv formatted this way
    """
    import os.path

    # Fill Header
    if with_header:
        header = ("Basename", "start", "end", "label", "prominence", "boundary")

    # Generate content
    lines = []
    for i in range(0, len(labels)):
        lines.append(("%s" %(os.path.splitext(os.path.basename(fname))[0] ),
                     "%.3f" %(float(labels[i][0]/frame_rate)),
                      "%.3f" %(float(labels[i][1]/frame_rate)),
                      labels[i][2],
                      #"%.3f" %(float(prominences[i][0]/frame_rate)),
                      "%.3f" %(prominences[i][1]),
                     "%.3f" %(boundaries[i][1])))

    logger.debug("Saving %s with following content:" % fname)
    if with_header:
        logger.debug(header)
    logger.debug(lines)

    import codecs
    with codecs.open(fname, "w", "utf-8") as prom_f:
        if with_header:
            prom_f.write(u"\t".join(header) + u"\n")

        for i in range(0,len(lines)):
            prom_f.write(u'\t'.join(lines[i])+u"\n")

def simplify(loma):

    simplified = []
    for l in loma:
        # align loma to it's position in the middle of the line
        pos =  l[int(len(l)/2.0)][0]
        strength = l[-1][1]
        simplified.append((pos,strength))
    return simplified

def get_prominences(pos_loma, labels, rate=1):

    max_word_loma = []
    loma = simplify(pos_loma)
    for (st, end, unit) in labels:
        st*=rate
        end*=rate
        word_loma = []
        for l in loma:
            if l[0] >=st and l[0]<=end:
                word_loma.append(l)# l[1])
        if len(word_loma)> 0:
            max_word_loma.append(sorted(word_loma, key=itemgetter(1))[-1])
        else:
            max_word_loma.append([st+(end-st)/2.0, 0.])

    return max_word_loma





# get strongest lines of minimum amplitude between adjacent words' max lines
def get_boundaries(max_word_loma,boundary_loma, labels):
    boundary_loma = simplify(boundary_loma)
    max_boundary_loma = []
    st = 0
    end=0
    for i in range(1, len(max_word_loma)):
        w_boundary_loma = []
        for l in boundary_loma:
            st = max_word_loma[i-1][0]
            end = max_word_loma[i][0]
            if l[0] >=st and l[0]<end:
                if l[1] > 0:
                    w_boundary_loma.append(l)

        if len(w_boundary_loma) > 0:
            max_boundary_loma.append(sorted(w_boundary_loma, key=itemgetter(1))[-1])
        else:
            max_boundary_loma.append([st+(end-st)/2, 0])

    # final boundary is not estimated
    max_boundary_loma.append((labels[-1][1],1))
    return max_boundary_loma




def get_peaks(params, threshold=-10):
    indices = []
    threshold = 0.001

    threshold = -100
    zc = np.where(np.diff(np.sign(np.diff(params))))[0]
    indices = (np.diff(np.sign(np.diff(params))) < 0).nonzero()[0] +1

    peaks = params[indices]
    return np.array([peaks[peaks>threshold], indices[peaks>threshold]])

def _get_parent(child_index, parent_diff, parent_indices):

    # at child peak index, follow the slope of parent scale upwards to find parent

    for i in range(0, len(parent_indices)):
        if (parent_indices[i] > child_index):
            if (parent_diff[int(child_index)] > 0):
                return parent_indices[i]
            else:
                if i > 0:
                    return parent_indices[i-1]
                else:
                    return parent_indices[0]

    if len(parent_indices) > 0:
        return parent_indices[-1]
    return None


# lines of maximum amplitude

# note: change this so that one level is done in one chunk, not one parent.

def get_loma(wavelet_matrix, scales, min_scale, max_scale):
    """Get the Line Of Maximum Amplitude (loma)
    """
    psize = 100.0
    min_peak = -10000.0 # minimum peak amplitude to consider. NOTE:this has no meaning unless scales normalized
    max_dist = 10 # how far in time to look for parent peaks. NOTE: frame rate and scale dependent, FIXME: how dependent?

    # get peaks from the first scale
    (peaks,indices) = get_peaks(wavelet_matrix[min_scale],min_peak)

    loma=dict()
    root=dict()
    for i in range(0,len(peaks)):
        loma[indices[i]]=[]

        # keep track of roots of each loma
        root[indices[i]] = indices[i]

    for i in range(min_scale+1, max_scale):
        max_dist = np.sqrt(scales[i])*4

	# find peaks in the parent scale
        (p_peaks,p_indices) = get_peaks(wavelet_matrix[i], min_peak)

        parents = dict(zip(p_indices, p_peaks))

        # find a parent for each child peak
        children = dict()
        for p in p_indices:
            children[p] = []

        parent_diff = np.diff(wavelet_matrix[i],1)
        for j in range(0,len(indices)):
            parent =_get_parent(indices[j], parent_diff, p_indices)
            if parent:
                if abs(parent-indices[j]) < max_dist and peaks[j] > min_peak:#  np.std(wavelet_matrix[i])*0.5:
                    children[parent].append([indices[j],peaks[j]])
        peaks=[];indices = []

        # for each parent, select max child

        for p in children:

            if len(children[p]) > 0:
		# maxi[0]: index
		# maxi[1]: peak height
                maxi = sorted(children[p], key=itemgetter(1))[-1]
                indices.append(p)
                peaks.append(maxi[1]+parents[p])

                #append child to correct loma
                loma[root[maxi[0]]].append([maxi[0],maxi[1]+parents[p]])
                root[p] = root[maxi[0]]

                # #plot for debugging
                # if fig:
                #     y = i-1
                #     size = maxi[1]+parents[p]
                #     fig.plot([maxi[0], p], [(i-2), y],
                #              linewidth=2+size, color=color,
                #              alpha=0.45, solid_capstyle='round')

    sorted_loma = []
    for k in sorted(loma.keys()):
        if  len(loma[k]) > 0:
            sorted_loma.append(loma[k])

    logger.debug(simplify(sorted_loma))
    return sorted_loma
