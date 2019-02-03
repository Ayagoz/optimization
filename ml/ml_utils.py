import numpy as np

from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .ExpTransformer import ExpTransformer


def hinge_loss_coef(y_pred, y_true):
    if np.min(y_true) == 0:
        y_copy = 2 * y_true.copy() - 1
    else:
        y_copy = np.copy(y_true)
    return np.where(y_pred * y_copy < 1, -y_copy, 0)

def MLE_l2_loss(y, log_proba, beta):
    return -np.sum(y * log_proba.T[1] + (1-y)*log_proba.T[0]) + np.sum(beta**2)

def find_pipeline_params(X, y, params, n_jobs=5, random_state=0, scaled=False, kernel=False, scoring='roc_auc',
                         n_splits=100):
    if scaled and kernel:
        pipe = Pipeline([('scale', StandardScaler(with_std=False)), ('kernel', ExpTransformer()),
                         ('ml', LogisticRegression(max_iter=10 ** 5))])
    elif not scaled and kernel:
        pipe = Pipeline([('kernel', ExpTransformer()), ('ml', LogisticRegression(n_jobs=1, max_iter=10 ** 5))])
    elif scaled and not kernel:
        pipe = Pipeline([('scale', StandardScaler(with_std=False)),
                         ('ml', LogisticRegression(n_jobs=1, max_iter=10 ** 5))])
    else:
        pipe = Pipeline([('ml', LogisticRegression(n_jobs=1, max_iter=10 ** 5))])

    cv = StratifiedShuffleSplit(n_splits=n_splits, random_state=random_state)
    gr = GridSearchCV(estimator=pipe, param_grid=params, scoring=scoring, n_jobs=n_jobs, cv=cv)

    gr.fit(X, y)

    return gr.best_params_


def adam_step(gt, mt, vt, t, beta1=0.9, beta2=0.999, eps=1e-8):
    mt1 = beta1 * mt + (1 - beta1) * gt
    vt1 = beta2 * vt + (1 - beta2) * gt ** 2
    hatm = mt1 / (1 - beta1 ** t)
    hatv = vt1 / (1 - beta2 ** t)
    return hatm / (np.sqrt(hatv) + eps), mt1, vt1


def expand_dims(a, ndim):
    shape = a.shape
    new_shape = shape + (1,) * ndim
    return a.reshape(new_shape)


def diff_K_exp_kernel_scaled(alpha, K_train, dK_train, K_test, dK_test):
    dK_mean = 1 / float(K_train.shape[-1]) * np.sum(K_train * dK_train, axis=0)
    return -alpha * (K_train * dK_train - dK_mean), -alpha * (K_test * dK_test - dK_mean)


def diff_K_exp_kernel(alpha, K_train, dK_train, K_test, dK_test):
    return -alpha * K_train * dK_train, -alpha * K_test * dK_test


def diff_K_scaled(dK_train, dK_test):
    dK_mean = 1 / float(dK_train.shape[-1]) * np.sum(dK_train, axis=0)
    return (dK_train - dK_mean), (dK_test - dK_mean)


def diff_J_kernel(alpha, K_train, dK_train, K_test, dK_test, ndim):
    return -alpha * expand_dims(K_train, ndim) * dK_train, \
                       -alpha * expand_dims(K_test, ndim) * dK_test



def diff_loss_by_J(dK_dJ_train, dK_dJ_test, K_train, K_test, y_train, y_test, proba_test,
                      proba_train, H, beta, ndim, C):

    # loss = -MLE + ||beta||^2
    # d loss /dJ =  - sum ((y_true - proba) * (beta^T * dx/dJ + d beta^T/ dJ * x)) + 2 beta * d beta /dJ
    # dx / dJ = dK_dJ
    # d beta/ dJ = (d^2 loss/d beta^2)^(-1) (d^2 loss/ d beta /d J)
    # (d^2 loss /d beta)^(-1) = H
    # d^2 loss /d beta /dJ = dX/dJ * (sigma - y_true) + X *(d sigma/dJ)
    # d sigma / dJ = (1-sigma) * sigma * beta^T *  dx/ dJ
    p_t = (proba_train - y_train)[None]
    s1s = proba_train * (1 - proba_train)

    dd_dJ = (dK_dJ_train * expand_dims(p_t, ndim)).sum(axis=0)+ \
            (expand_dims(K_train, ndim) * (expand_dims((s1s * beta).T, ndim) * dK_dJ_train).sum(axis=0)[None]).sum(axis=1)

    dbeta_dJ = (expand_dims(H, ndim) * dd_dJ).sum(axis=1)
    
    dbeta_dx_dJ = (dK_dJ_test * expand_dims(beta, ndim)).sum(axis=1) + (expand_dims(K_test, ndim)* dbeta_dJ).sum(axis=1)
    
    dxdJ = - np.sum(expand_dims(y_test - proba_test, ndim) * dbeta_dx_dJ, axis=0).sum(axis=0) + (2 * C * expand_dims(beta.T, ndim) *  dbeta_dJ).sum(axis=(0,1))

    return dxdJ

def diff_loss_by_a_b(dK_da_train, dK_da_test, dK_db_train, dK_db_test, K_test, K_train,
                                            y_train, y_test, proba_test, proba_train, beta, C):
    '''
    for loss(beta) = -MLE + ||beta||^2 minimization
    '''
    #d beta/ da = (d^2 loss/d beta^2)^(-1) (d^2 loss/ d beta /d a)

    # sigma *(1-sigma)
    s1s = (1 - proba_train) * proba_train
    # d^2 loss / d beta^2 = (X^T B X + 2E) = H
    # B = diag(sigma_i*(1-sigma_i))
    H = np.linalg.pinv(- K_train.dot(np.diag(s1s)).dot(K_train) + 2 * np.eye(K_train.shape[0]))

    # sigma - y_true
    p_t = (proba_train - y_train)[None]

    # d^2 loss /d beta/da = d( X^T (sigma - y_true)) / d a =
    # = dX/da * (sigma - y_true) + X *(d sigma/da)
    # d sigma / da = (1-sigma) * sigma * beta^T * dx/da
    # suppose that beta is constant, so d beta/da = 0

    dd_da = dK_da_train.dot(p_t.T) + K_train.dot((beta * s1s).dot(dK_da_train).T)
    dd_db = dK_db_train.dot(p_t.T) + K_train.dot((beta * s1s).dot(dK_db_train).T)

    #d beta / da = H * dd_da
    dbeta_da = - H.dot(dd_da)
    dbeta_db = - H.dot(dd_db)

    # dbeta/da = d argmin(MLE + l2(beta))/ da = (d^2(MLE + l2(beta))/dbeta^2)^(-1).(d(MLE + l2(beta))^2/da/dbeta)
    # H = (d^2(MLE + l2(beta))/dbeta^2)^(-1), DD_da = (d(MLE + l2(beta))^2/da/dbeta)

    #d loss / da = - sum ((y_true - proba) * (beta^T * dx/da + d beta^T/ da * x)) + 2 beta * d beta /da
    dbeta_dx_da = (dK_da_test.dot(beta.T) + K_test.dot(dbeta_da))
    dbeta_dx_db = (dK_db_test.dot(beta.T) + K_test.dot(dbeta_db))

    dxda = - np.sum((y_test - proba_test) * dbeta_dx_da) + 2 * C * beta.dot(dbeta_da)
    dxdb = - np.sum((y_test - proba_test) * dbeta_dx_db) + 2 * C * beta.dot(dbeta_db)

    return np.sum(dxda), np.sum(dxdb), H
