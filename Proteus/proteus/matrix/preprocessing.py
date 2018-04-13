
import numpy as np
import os
import threading
import warnings
from scipy.linalg import svd
from sklearn.base import BaseEstimator, TransformerMixin

class ImageDataGenerator(object):
    """Generate minibatches of image data with real-time data augmentation.
    # Arguments
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        zca_whitening: apply ZCA whitening.

        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        preprocessing_function: function that will be implied on each input.
            The function will run before any other modification on it.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
    """

    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 fill_mode='nearest',
                 cval=0.,
                 rescale=None,
                 preprocessing_function=None,
                 channel_axis=3):
        self.featurewise_center = featurewise_center
        self.samplewise_center = samplewise_center
        self.featurewise_std_normalization = featurewise_std_normalization
        self.samplewise_std_normalization = samplewise_std_normalization
        self.zca_whitening = zca_whitening
        self.fill_mode = fill_mode
        self.cval = cval
        self.rescale = rescale
        self.preprocessing_function = preprocessing_function
        self.mean = None
        self.std = None
        self.principal_components = None
        self.channel_axis = channel_axis

    def standardize(self, x):
        if self.preprocessing_function:
            x = self.preprocessing_function(x)
        if self.rescale:
            x *= self.rescale
        # x is a single image, so it doesn't have image number at index 0
        if self.samplewise_center:
            x -= np.mean(x, axis=0, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, axis=0, keepdims=True) + 1e-7)

        if self.featurewise_center:
            if self.mean is not None:
                x -= self.mean
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_center`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.featurewise_std_normalization:
            if self.std is not None:
                x /= (self.std + 1e-7)
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_std_normalization`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.zca_whitening:
            if self.principal_components is not None:
                flatx = np.reshape(x, (x.size))
                whitex = np.dot(flatx, self.principal_components)
                x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`zca_whitening`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        return x

    def fit(self, x,
            augment=False,
            rounds=1,
            seed=None):
        """Required for featurewise_center, featurewise_std_normalization
        and zca_whitening.
        # Arguments
            x: Numpy array, the data to fit on. Should have rank 4.
                In case of grayscale data,
                the channels axis should have value 1, and in case
                of RGB data, it should have value 3.
            augment: Whether to fit on randomly augmented samples
            rounds: If `augment`,
                how many augmentation passes to do over the data
            seed: random seed.
        # Raises
            ValueError: in case of invalid input `x`.
        """
        x = np.asarray(x)
        if x.ndim != 4:
            raise ValueError('Input to `.fit()` should have rank 4. '
                             'Got array with shape: ' + str(x.shape))

        '''
        if x.shape[self.channel_axis] not in {1, 3, 4}:
            raise ValueError(
                'Expected input to be images (as Numpy array) '
                'following the dimension ordering convention "' + self.dim_ordering + '" '
                                                                                      '(channels on axis ' + str(
                    self.channel_axis) + '), i.e. expected '
                                         'either 1, 3 or 4 channels on axis ' + str(self.channel_axis) + '. '
                                                                                                         'However, it was passed an array with shape ' + str(
                    x.shape) +
                ' (' + str(x.shape[self.channel_axis]) + ' channels).')

        '''

        if seed is not None:
            np.random.seed(seed)

        x = np.copy(x)
        if augment:
            ax = np.zeros(tuple([rounds * x.shape[0]] + list(x.shape)[1:]))
            for r in range(rounds):
                for i in range(x.shape[0]):
                    ax[i + r * x.shape[0]] = self.random_transform(x[i])
            x = ax

        if self.featurewise_center:
            self.mean = np.mean(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.mean = np.reshape(self.mean, broadcast_shape)
            x -= self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.std = np.reshape(self.std, broadcast_shape)
            x /= (self.std + K.epsilon())

        if self.zca_whitening:
            self.bias = 0.1
            self.copy = True
            self.n_components = 5
            flat_x = np.reshape(x, (x.shape[3], x.shape[0] * x.shape[1] * x.shape[2]))
            X = np.array(flat_x, copy=self.copy)
            n_samples, n_features = X.shape
            self.mean_ = np.mean(X, axis=0)
            X -= self.mean_
            U, S, VT = svd(X, full_matrices=False)

            print U.shape,S.shape,VT.shape
            components = np.dot((VT.T * np.sqrt(1.0 / (S ** 2 + self.bias)))[:self.n_components, :], VT)
            self.principal_components_ = components



            #sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
            #u, s, v = linalg.svd(sigma)
            #self.principal_components = np.dot(np.dot(u, np.diag(1. / np.sqrt(s + 10e-7))), u.T)


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import gen_batches
from scipy.linalg import eigh
from scipy.linalg import svd
import numpy as np

# From sklearn master
def svd_flip(u, v, u_based_decision=True):
    """Sign correction to ensure deterministic output from SVD.

    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.

    Parameters
    ----------
    u, v : ndarray
        u and v are the output of `linalg.svd` or
        `sklearn.utils.extmath.randomized_svd`, with matching inner dimensions
        so one can compute `np.dot(u * s, v)`.

    u_based_decision : boolean, (default=True)
        If True, use the columns of u as the basis for sign flipping. Otherwise,
        use the rows of v. The choice of which variable to base the decision on
        is generally algorithm dependent.


    Returns
    -------
    u_adjusted, v_adjusted : arrays with the same dimensions as the input.

    """
    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, xrange(u.shape[1])])
        u *= signs
        v *= signs[:, np.newaxis]
    else:
        # rows of v, columns of u
        max_abs_rows = np.argmax(np.abs(v), axis=1)
        signs = np.sign(v[xrange(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, np.newaxis]
    return u, v


def _batch_mean_variance_update(X, old_mean, old_variance, old_sample_count):
    """Calculate an average mean update and a Youngs and Cramer variance update.

    From the paper "Algorithms for computing the sample variance: analysis and
    recommendations", by Chan, Golub, and LeVeque.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data to use for variance update

    old_mean : array-like, shape: (n_features,)

    old_variance : array-like, shape: (n_features,)

    old_sample_count : int

    Returns
    -------
    updated_mean : array, shape (n_features,)

    updated_variance : array, shape (n_features,)

    updated_sample_count : int

    References
    ----------
    T. Chan, G. Golub, R. LeVeque. Algorithms for computing the sample variance:
        recommendations, The American Statistician, Vol. 37, No. 3, pp. 242-247

    """
    new_sum = X.sum(axis=0)
    new_variance = X.var(axis=0) * X.shape[0]
    old_sum = old_mean * old_sample_count
    n_samples = X.shape[0]
    updated_sample_count = old_sample_count + n_samples
    partial_variance = old_sample_count / (n_samples * updated_sample_count) * (
        n_samples / old_sample_count * old_sum - new_sum) ** 2
    unnormalized_variance = old_variance * old_sample_count + new_variance + \
        partial_variance
    return ((old_sum + new_sum) / updated_sample_count,
            unnormalized_variance / updated_sample_count,
            updated_sample_count)


class _CovZCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, bias=.1, copy=True):
        self.n_components = n_components
        self.bias = bias
        self.copy = copy

    def fit(self, X, y=None):
        if self.copy:
            X = np.array(X, copy=self.copy)
        n_samples, n_features = X.shape
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_
        U, S, VT = svd(np.dot(X.T, X) / n_samples, full_matrices=False)
        components = np.dot(VT.T * np.sqrt(1.0 / (S + self.bias)), VT)
        self.components_ = components[:self.n_components]
        return self

    def transform(self, X):
        if self.copy:
            X = np.array(X, copy=self.copy)
        X -= self.mean_
        X_transformed = np.dot(X, self.components_.T)
        return X_transformed


class ZCA(BaseEstimator, TransformerMixin):
    """
    Identical to CovZCA up to scaling due to lack of division by n_samples
    S ** 2 / n_samples should correct this but components_ come out different
    though transformed examples are identical.
    """
    def __init__(self, n_components=None, bias=.1, copy=True):
        self.n_components = n_components
        self.bias = bias
        self.copy = copy

    def fit(self, X, y=None):
        if self.copy:
            X = np.array(X, copy=self.copy)
        n_samples, n_features = X.shape
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_
        U, S, VT = svd(X, full_matrices=False)
        self.explained_variance_ = S
        tmp_ = VT.T * np.sqrt(1.0 / (S ** 2 + self.bias))
        print tmp_.shape, VT.shape, S.shape
        self.components_ = np.dot(tmp_[:self.n_components, :], VT)
        return self

    def transform(self, X):
        if self.copy:
            X = np.array(X, copy=self.copy)
            X = np.copy(X)
        X -= self.mean_
        X_transformed = np.dot(X, self.components_.T)
        return X_transformed

    def inverse_transform(self, X_transformed):
        if self.copy:
            X_transformed = np.array(X_transformed, copy=self.copy)
            X_transformed = np.copy(X_transformed)

        return np.dot(X_transformed, self.components_) #+ self.mean_

class IncrementalZCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, batch_size=None, bias=.1,
                 scale_by=1., copy=True):
        self.n_components = n_components
        self.batch_size = batch_size
        self.bias = bias
        self.scale_by = scale_by
        self.copy = copy
        self.scale_by = float(scale_by)
        self.n_samples_seen_ = 0.
        self.mean_ = None
        self.var_ = None
        self.components_ = None

    def fit(self, X, y=None):
        self.n_samples_seen_ = 0.
        self.mean_ = None
        self.var_ = None
        self.components_ = None
        n_samples, n_features = X.shape
        if self.batch_size is None:
            self.batch_size_ = 5 * n_features
        else:
            self.batch_size_ = self.batch_size
        for batch in gen_batches(n_samples, self.batch_size_):
            self.partial_fit(X[batch])
        return self

    def partial_fit(self, X):
        if self.copy:
            X = np.array(X, copy=self.copy)
            X = np.copy(X)
        n_samples, n_features = X.shape
        self.n_components_ = self.n_components
        X /= self.scale_by
        if self.components_ is None:
            # This is the first pass through partial_fit
            self.n_samples_seen_ = 0.
            col_var = X.var(axis=0)
            col_mean = X.mean(axis=0)
            X -= col_mean
            U, S, V = svd(X, full_matrices=False)
            U, V = svd_flip(U, V, u_based_decision=False)
        else:
            col_batch_mean = X.mean(axis=0)
            col_mean, col_var, n_total_samples = _batch_mean_variance_update(
                X, self.mean_, self.var_, self.n_samples_seen_)
            X -= col_batch_mean
            # Build matrix of combined previous basis and new data
            correction = np.sqrt((self.n_samples_seen_ * n_samples)
                                 / n_total_samples)
            mean_correction = correction * (self.mean_ - col_batch_mean)
            X_combined = np.vstack((self.singular_values_.reshape((-1, 1)) *
                                    self.components_,
                                    X,
                                    mean_correction))
            U, S, V = svd(X_combined, full_matrices=False)
            U, V = svd_flip(U, V, u_based_decision=False)

        self.n_samples_seen_ += n_samples
        self.components_ = V[:self.n_components_]
        self.singular_values_ = S[:self.n_components_]
        self.mean_ = col_mean
        self.var_ = col_var
        self.zca_components_ = np.dot(self.components_.T * np.sqrt(1.0 / (self.singular_values_ ** 2 + self.bias)), self.components_)

    def transform(self, X):
        if self.copy:
            X = np.array(X, copy=self.copy)
            X = np.copy(X)
        X /= self.scale_by
        X -= self.mean_
        X_transformed = np.dot(X, self.zca_components_.T)
        return X_transformed
