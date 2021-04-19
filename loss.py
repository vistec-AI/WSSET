import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

def emd_batch(batch,EMD):
    matrix = np.zeros((batch.shape[0],batch.shape[0]))
    for x in range(batch.shape[0]):
        idx1=batch[x]
        for y in range(batch.shape[0]):
            idx2=batch[y]
            if idx1>idx2:
                e=float(EMD[idx2][idx1])
            else:
                e=float(EMD[idx1][idx2])
            matrix[x][y]=e
    return matrix

#direct approximate EMD loss
def emd_loss(y_true, y_pred,EMD):
    sum=pairwise_distance(y_pred)
    emd=emd_batch(y_true,EMD).astype(np.float32)
    diff=tf.math.square(tf.math.subtract(emd,sum)+1e-12)
    diff=tf.math.sqrt(diff)
    total = tf.reduce_sum(diff)
    loss = tf.math.truediv(total,float(y_true.shape[0]**2),)
    print(loss)
    return loss

@tf.function
def pairwise_distance(feature, squared=False):
    """Computes the pairwise distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2
    Args:
      feature: 2-D Tensor of size [number of data, feature dimension].
      squared: Boolean, whether or not to square the pairwise distances.
    Returns:
      pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """
    pairwise_distances_squared = tf.math.add(
        tf.math.reduce_sum(tf.math.square(feature), axis=[1], keepdims=True),
        tf.math.reduce_sum(
            tf.math.square(tf.transpose(feature)), axis=[0], keepdims=True
        ),
    ) - 2.0 * tf.matmul(feature, tf.transpose(feature))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = tf.math.maximum(pairwise_distances_squared, 0.0)
    # Get the mask where the zero distances are at.
    error_mask = tf.math.less_equal(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = tf.math.sqrt(
            pairwise_distances_squared
            + tf.cast(error_mask, dtype=tf.dtypes.float32) * 1e-16
        )

    # Undo conditionally adding 1e-16.
    pairwise_distances = tf.math.multiply(
        pairwise_distances,
        tf.cast(tf.math.logical_not(error_mask), dtype=tf.dtypes.float32),
    )

    num_data = tf.shape(feature)[0]
    # Explicitly set diagonals to zero.
    mask_offdiagonals = tf.ones_like(pairwise_distances) - tf.linalg.diag(
        tf.ones([num_data])
    )
    pairwise_distances = tf.math.multiply(pairwise_distances, mask_offdiagonals)
    return pairwise_distances

def _masked_minimum(data, mask, dim=1):
    """Computes the axis wise minimum over chosen elements.
    Args:
      data: 2-D float `Tensor` of size [n, m].
      mask: 2-D Boolean `Tensor` of size [n, m].
      dim: The dimension over which to compute the minimum.
    Returns:
      masked_minimums: N-D `Tensor`.
        The minimized dimension is of size 1 after the operation.
    """
    axis_maximums = tf.math.reduce_max(data, dim, keepdims=True)
    masked_minimums = (
        tf.math.reduce_min(
            tf.math.multiply(data - axis_maximums, mask), dim, keepdims=True
        )
        + axis_maximums
    )
    return masked_minimums


def _masked_maximum(data, mask, dim=1):
    """Computes the axis wise maximum over chosen elements.
    Args:
      data: 2-D float `Tensor` of size [n, m].
      mask: 2-D Boolean `Tensor` of size [n, m].
      dim: The dimension over which to compute the maximum.
    Returns:
      masked_maximums: N-D `Tensor`.
        The maximized dimension is of size 1 after the operation.
    """
    axis_minimums = tf.math.reduce_min(data, dim, keepdims=True)
    masked_maximums = (
        tf.math.reduce_max(
            tf.math.multiply(data - axis_minimums, mask), dim, keepdims=True
        )
        + axis_minimums
    )
    return masked_maximums

def inverse_gaussian(data):
    std = tf.math.reduce_std(data,axis=1,keepdims=True)
    a= tf.math.square(data) #(x-mu)^2
    a = tf.math.scalar_mul(-1.0,a)
    std = tf.math.scalar_mul(12.0,std)
    sqstd = tf.math.square(std) #std^2
    b=tf.math.scalar_mul(2.0,sqstd) #2*std^2
    a= tf.math.truediv(a,b)
    a = tf.math.exp(a)
    return a

#Our main WSSET loss
def WSSET_loss(y_true,y_pred,w,treshold,EMD,margin):

    batch_size = y_pred.shape[0]
    pdist_matrix=pairwise_distance(y_pred,squared=True)
    emd=tf.cast(emd_batch(y_true,EMD), dtype=tf.dtypes.float32)
    if w == 'gaussian':
        weight = inverse_gaussian(emd)
    elif w == '1':
        weight = 1
    elif w == 'emd':
        weight = emd
    
    weighted_emd = tf.math.multiply(pdist_matrix,weight)
    sort = tf.sort(emd)
    sort = tf.slice(sort,[0,2],[batch_size,1])
    tresh = tfp.stats.percentile(emd,treshold,keep_dims=True)
    adjacency1 = tf.math.less(emd, sort)
    adjacency2 = tf.math.less(emd, tresh)
    adjacency = tf.math.logical_and(adjacency1,adjacency2)
    adjacency_not = tf.math.logical_not(adjacency)
    pdist_matrix_tile = tf.tile(emd, [batch_size, 1])
    pdist_matrix_tile2 = tf.tile(weighted_emd, [batch_size, 1])
    mask = tf.math.logical_and(
        tf.tile(adjacency_not, [batch_size, 1]),
        tf.math.greater(
            pdist_matrix_tile2, tf.reshape(tf.transpose(weighted_emd), [-1, 1])
        ),
    )
    mask_final = tf.reshape(
        tf.math.greater(
            tf.math.reduce_sum(
                tf.cast(mask, dtype=tf.dtypes.float32), 1, keepdims=True
            ),
            0.0,
        ),
        [batch_size, batch_size],
    )
    mask_final = tf.transpose(mask_final)
    mask_positives = tf.cast(adjacency, dtype=tf.dtypes.float32) - tf.linalg.diag(
        tf.ones([batch_size])
    )
    mask_negatives = tf.cast(adjacency_not, dtype=tf.dtypes.float32)
    mask = tf.cast(mask, dtype=tf.dtypes.float32)
    
    mask_pos_wo = tf.cast(adjacency, dtype=tf.dtypes.float32)
    
    negatives_outside = tf.reshape(
        _masked_minimum(pdist_matrix_tile2, mask), [batch_size, batch_size]
    )
    negatives_outside = tf.transpose(negatives_outside)
    
    negatives_inside = tf.math.multiply(tf.tile(
        _masked_maximum(weighted_emd, mask_negatives), [1, batch_size]
    ),mask_pos_wo)

    semi_hard_negatives = tf.where(mask_final, negatives_outside, negatives_inside)
    loss_mat = tf.math.add(margin, pdist_matrix - semi_hard_negatives)
    num_positives = tf.math.reduce_sum(mask_positives)
    loss = tf.math.truediv(tf.math.reduce_sum(tf.math.maximum(tf.math.multiply(loss_mat,mask_positives),0.0)),num_positives)
    print(loss)
    return loss

