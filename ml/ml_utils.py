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


def diff_exp_K(alpha, K_train, dK_train, K_test, dK_test, scaled=False,
               kernel=False, template=False, ndim=None):
    if scaled:
        # because scaled exp_K^ = exp_K - np.mean(exp_K, axis=0)
        # dexp_K^/da = dexp_K/da - mean(dexp_K/da, axis=0)
        # dexp_K/da = exp_K * (-alpha) * dK/da

        if kernel:
            dK_mean = 1 / float(K_train.shape[-1]) * np.sum(K_train * dK_train, axis=0)
            return -alpha * (K_train * dK_train - dK_mean), -alpha * (K_test * dK_test - dK_mean)
        else:
            dK_mean = 1 / float(K_train.shape[-1]) * np.sum(dK_train, axis=0)
            return (dK_train - dK_mean), (dK_test - dK_mean)

    else:
        if template:
            if kernel:
                return -alpha * expand_dims(K_train, ndim) * dK_train, \
                       -alpha * expand_dims(K_test, ndim) * dK_test
            else:
                return dK_train, dK_test
        else:
            if kernel:
                return -alpha * K_train * dK_train, -alpha * K_test * dK_test
            else:
                return dK_train, dK_test


def diff_template(y_test, proba_test, K_train, dJ_train, K_test, dJ_test, H, s1s, beta,
                  p_train, alpha, ndim, kernel=False):
    dexp_dJ_train, dexp_dJ_test = diff_exp_K(alpha=alpha, K_train=K_train, dK_train=dJ_train,
                                             K_test=K_test, dK_test=dJ_test, scaled=False,
                                             kernel=kernel, template=True, ndim=ndim)

    dd_dJ = (dexp_dJ_train * expand_dims(p_train, ndim)).sum(axis=0) + \
            (K_train.dot(expand_dims(s1s * beta, ndim).dot(dexp_dJ_train))).sum(axis=0)

    dbeta_dJ = - expand_dims(H, ndim).dot(dd_dJ)
    dbeta_dx_dJ = expand_dims(beta, ndim).dot(dexp_dJ_test) + dbeta_dJ.dot(dJ_test)

    dxdJ = np.zeros(dJ_train.shape[-ndim:])

    for i in range(K_test.shape[0]):
        dxdJ += - (y_test - proba_test) * dbeta_dx_dJ[i] + 2 * expand_dims(beta, ndim).dot(dbeta_dJ)

    return dxdJ


def diff_hinge_loss_lr(beta, proba_test, proba_train, y_test, y_train, exp_K_test,
                       exp_K_train, da_train, db_train, da_test, db_test, alpha=None,
                       dJ_train=None, dJ_test=None, scaled=False, kernel=False,
                       with_template=False, ndim=None):
    '''
    for MLE + ||beta||^2 maximization
    (try another for Entropy?)
    '''
    # sigma *(1-sigma)
    s1s = (1 - proba_train) * proba_train

    # H =  (X^T * B * X + 2 E)^(-1), B = diag(sigma_i*(1-sigma_i))
    H = np.linalg.pinv(-exp_K_train.dot(np.diag(s1s)).dot(exp_K_train) + 2 * np.eye(exp_K_train.shape[0]))

    dexp_da_train, dexp_da_test = diff_exp_K(alpha=alpha, K_train=exp_K_train, dK_train=da_train,
                                             K_test=exp_K_test, dK_test=da_test, scaled=scaled,
                                             kernel=kernel, template=False)

    dexp_db_train, dexp_db_test = diff_exp_K(alpha=alpha, K_train=exp_K_train, dK_train=db_train,
                                             K_test=exp_K_test, dK_test=db_test, scaled=scaled,
                                             kernel=kernel, template=False)

    # sigma - y_true
    p_t = (proba_train - y_train)[None, :]

    # d argmin(MLE + l2(beta))/d beta/da
    # dX/da * (sigma - y_true) + X *(dsigma/da)
    # suppose that beta is constant, so dbeta/da = 0

    dd_da = dexp_da_train.dot(p_t.T) + exp_K_train.dot((beta * s1s).dot(dexp_da_train).T)
    dd_db = dexp_db_train.dot(p_t.T) + exp_K_train.dot((beta * s1s).dot(dexp_db_train).T)

    dbeta_da = - H.dot(dd_da)
    dbeta_db = - H.dot(dd_db)
    # d hinge_loss(y^)/da = -t* dy^/da *I(t*y < 1), I - indicator function
    # dy^/da = dbeta/da * X + beta * dX/da
    # dbeta/da = d argmin(MLE + l2(beta))/ da = (d^2(MLE + l2(beta))/dbeta^2)^(-1).(d(MLE + l2(beta))^2/da/dbeta)
    # H = (d^2(MLE + l2(beta))/dbeta^2)^(-1), DD_da = (d(MLE + l2(beta))^2/da/dbeta)

    dbeta_dx_da = (dexp_da_test.dot(beta.T) + exp_K_test.dot(dbeta_da))
    dbeta_dx_db = (dexp_db_test.dot(beta.T) + exp_K_test.dot(dbeta_db))

    dxda = - np.sum((y_test - proba_test) * dbeta_dx_da) + 2 * beta.dot(dbeta_da)
    dxdb = - np.sum((y_test - proba_test) * dbeta_dx_db) + 2 * beta.dot(dbeta_db)

    if with_template:
        if dJ_train is None and dJ_test is None:
            raise TypeError('No template derivatives in mode with_template!')

        dJ = diff_template(y_test=y_test, proba_test=proba_test,
                           K_train=exp_K_train, dJ_trian=dJ_train,
                           K_test=exp_K_test, dJ_test=dJ_test,
                           H=H, s1s=s1s, beta=beta, p_train=p_t, alpha=alpha, ndim=ndim)
        return np.sum(dxda), np.sum(dxdb), np.array(dJ)

    return np.sum(dxda), np.sum(dxdb)
