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


def find_pipeline_params(X, y, params, n_jobs=5, random_state=0, scaled=False, scoring='roc_auc', n_splits=100):
    if scaled:
        pipe = Pipeline([('scale', StandardScaler(with_std=False)), ('kernel', ExpTransformer()),
                         ('ml', LogisticRegression(max_iter=10 ** 5))])
    else:
        pipe = Pipeline([('kernel', ExpTransformer()), ('ml', LogisticRegression(n_jobs=1, max_iter=10 ** 5))])

    cv = StratifiedShuffleSplit(n_splits, random_state=random_state)
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


def diff_exp_K(alpha, K_train, dK_train, K_test, dK_test, scaled=False, template=False, ndim=None):
    if scaled:
        # because scaled exp_K^ = exp_K - np.mean(exp_K, axis=0)
        # dexp_K^/da = dexp_K/da - mean(dexp_K/da, axis=0)
        # dexp_K/da = exp_K * (-alpha) * dK/da

        dK_mean = 1 / float(K_train.shape[-1]) * np.sum(K_train * dK_train, axis=0)
        return -alpha * (K_train * dK_train - dK_mean), -alpha * (K_test * dK_test - dK_mean)
    else:
        if template:
            return -alpha * expand_dims(K_train, ndim) * dK_train, -alpha * expand_dims(K_test, ndim) * dK_test

        else:
            return -alpha * K_train * dK_train, -alpha * K_test * dK_test


def diff_template(K_train, dJ_train, K_test, dJ_test, H, s1s, beta, loss_coef, p_train, alpha, ndim):
    dexp_dJ_train, dexp_dJ_test = diff_exp_K(alpha, K_train, dJ_train, K_test, dJ_test, False, True, ndim)

    dd_dJ = (expand_dims(p_train.T, ndim) * dexp_dJ_train).sum(axis=0) + \
            (expand_dims((beta * s1s).dot(K_train).T, ndim) * dexp_dJ_train).sum(axis=0)

    dxdJ = np.zeros(dJ_train.shape[-ndim:])

    for i in range(K_test.shape[0]):
        if loss_coef[i] != 0:
            dxdJ += - loss_coef[i] * ((expand_dims(K_test.dot(H), ndim)[i] * dd_dJ).sum(axis=0) +
                                      (expand_dims(beta, ndim) * dexp_dJ_test[i, ...]).sum(axis=(0, 1)))

    return dxdJ


def diff_hinge_loss_lr(loss_coef, beta, proba_train, y_train, exp_K_test,
                       exp_K_train, da_train, db_train, da_test, db_test, alpha,
                       dJ_train=None, dJ_test=None, scaled=False, with_template=False, ndim=None):
    '''
    for MLE + ||beta||^2 maximization
    (try another for Entropy?)
    '''
    # sigma *(1-sigma)
    s1s = (1 - proba_train) * proba_train

    # H =  (X^T * B * X + 2 E)^(-1), B = diag(sigma_i*(1-sigma_i))
    H = np.linalg.pinv(exp_K_train.dot(np.diag(s1s)).dot(exp_K_train) + 2 * np.eye(exp_K_train.shape[0]))

    dexp_da_train, dexp_da_test = diff_exp_K(alpha, exp_K_train, da_train, exp_K_test, da_test, scaled)
    dexp_db_train, dexp_db_test = diff_exp_K(alpha, exp_K_train, db_train, exp_K_test, db_test, scaled)

    # sigma - y_true
    p_t = (proba_train - y_train)[None, :]

    # d argmin(MLE + l2(beta))/d beta/da
    # dX/da * (sigma - y_true) + X *(dsigma/da)
    # suppose that beta is constant, so dbeta/da = 0

    dd_da = p_t.dot(dexp_da_train) + ((beta * s1s).dot(exp_K_train).dot(dexp_da_train))
    dd_db = p_t.dot(dexp_db_train) + ((beta * s1s).dot(exp_K_train).dot(dexp_db_train))

    # d hinge_loss(y^)/da = -t* dy^/da *I(t*y < 1), I - indicator function
    # dy^/da = dbeta/da * X + beta * dX/da
    # dbeta/da = d argmin(MLE + l2(beta))/ da = (d^2(MLE + l2(beta))/dbeta^2)^(-1).(d(MLE + l2(beta))^2/da/dbeta)
    # H = (d^2(MLE + l2(beta))/dbeta^2)^(-1), DD_da = (d(MLE + l2(beta))^2/da/dbeta)

    dxda = -loss_coef * (exp_K_test.dot(H.dot(dd_da.T)) + dexp_da_test.dot(beta.T))
    dxdb = -loss_coef * (exp_K_test.dot(H.dot(dd_db.T)) + dexp_db_test.dot(beta.T))


    if with_template:
        if dJ_train is None and dJ_test is None:
            raise TypeError('No template derivatives in mode with_template!')
        
        dJ = diff_template(exp_K_train, dJ_train, exp_K_test, dJ_test, H, s1s, beta, loss_coef, p_t, alpha, ndim)
        return np.sum(dxda), np.sum(dxdb), np.array(dJ)

    return np.sum(dxda), np.sum(dxdb)
