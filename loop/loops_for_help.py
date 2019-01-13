import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


from RegOptim.ml.ml_utils import diff_K_exp_kernel, diff_J_kernel
from RegOptim.ml.ml_utils import diff_loss_by_a_b, diff_loss_by_J
from RegOptim.ml.ml_utils import MLE_l2_loss

def count_grads_kernel_template(exp_K, da, db, dJ, y, params, n_splits=10, ndim=3, random_state=0):
    roc_aucs = []
    grads_a, grads_b = [], []
    grads_J = []
    losses = []

    cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=random_state)

    for idx_train, idx_test in cv.split(exp_K, y):

        exp_train, exp_test = exp_K[np.ix_(idx_train, idx_train)], exp_K[np.ix_(idx_test, idx_train)]
        y_train, y_test = y[idx_train], y[idx_test]

        lr = LogisticRegression(n_jobs=1, max_iter=10 ** 5)
        lr.set_params(**{'C': params['ml__C']})

        lr.fit(exp_train, y_train)

        proba_test = lr.predict_proba(exp_test).T[1]
        proba_train = lr.predict_proba(exp_train).T[1]
        log_proba = lr.predict_log_proba(exp_test)

        roc_aucs.append(roc_auc_score(y_test, proba_test))
        losses.append(MLE_l2_loss(y_test, log_proba, lr.coef_))

        da_train, da_test = diff_K_exp_kernel(alpha=params['kernel__gamma'],
                                              K_train=exp_train, dK_train=da[np.ix_(idx_train, idx_train)],
                                              K_test=exp_test,  dK_test=da[np.ix_(idx_test, idx_train)])

        db_train, db_test = diff_K_exp_kernel(alpha=params['kernel__gamma'],
                                              K_train=exp_train, dK_train=db[np.ix_(idx_train, idx_train)],
                                              K_test=exp_test, dK_test=db[np.ix_(idx_test, idx_train)])

        grad_a, grad_b, H = diff_loss_by_a_b(dK_da_train=da_train, dK_da_test=da_test,
                                             dK_db_train=db_train, dK_db_test=db_test,
                                             K_train=exp_train, K_test=exp_test,
                                             y_train=y_train, y_test=y_test,
                                             proba_train=proba_train, proba_test=proba_test,
                                             beta=lr.coef_
                                             )

        dJ_train, dJ_test = diff_J_kernel(alpha=params['kernel__gamma'],
                                          K_train=exp_train, dK_train=dJ[np.ix_(idx_train,idx_train)],
                                          K_test=exp_test, dK_test=dJ[np.ix_(idx_test, idx_train)], ndim=ndim)

        grad_J = diff_loss_by_J(dK_dJ_train=dJ_train, dK_dJ_test=dJ_test,
                                K_train=exp_train, K_test=exp_test,
                                y_train=y_train, y_test=y_test,
                                proba_train=proba_train, proba_test=proba_test,
                                H=H, beta=lr.coef_, ndim=ndim)

        grads_a.append(grad_a)
        grads_b.append(grad_b)
        grads_J.append(grad_J)

    return np.mean(grads_a), np.mean(grads_b), np.mean(grads_J, axis=0), np.mean(roc_aucs), np.mean(losses)


def count_grads_kernel(exp_K, da, db, y, params, n_splits=10, ndim=3, random_state=0):
    roc_aucs = []
    grads_a, grads_b = [], []
    losses = []

    cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=random_state)

    for idx_train, idx_test in cv.split(exp_K, y):

        exp_train, exp_test = exp_K[np.ix_(idx_train, idx_train)], exp_K[np.ix_(idx_test, idx_train)]
        y_train, y_test = y[idx_train], y[idx_test]

        lr = LogisticRegression(n_jobs=1, max_iter=10 ** 5)
        lr.set_params(**{'C': params['ml__C']})
        lr.fit(exp_train, y_train)

        proba_test = lr.predict_proba(exp_test).T[1]
        proba_train = lr.predict_proba(exp_train).T[1]
        log_proba = lr.predict_log_proba(exp_test)

        loss = -np.sum(y_test * log_proba.T[1] + (1-y_test)*log_proba.T[0]) + np.sum(lr.coef_**2)

        roc_aucs.append(roc_auc_score(y_test, proba_test))
        losses.append(loss)

        da_train, da_test = diff_K_exp_kernel(alpha=params['kernel__gamma'],
                                              K_train=exp_train, dK_train=da[np.ix_(idx_train, idx_train)],
                                              K_test=exp_test,  dK_test=da[np.ix_(idx_test, idx_train)])

        db_train, db_test = diff_K_exp_kernel(alpha=params['kernel__gamma'],
                                              K_train=exp_train, dK_train=db[np.ix_(idx_train, idx_train)],
                                              K_test=exp_test, dK_test=db[np.ix_(idx_test, idx_train)])

        grad_a, grad_b, H = diff_loss_by_a_b(dK_da_train=da_train, dK_da_test=da_test,
                                             dK_db_train=db_train, dK_db_test=db_test,
                                             K_train=exp_train, K_test=exp_test,
                                             y_train=y_train, y_test=y_test,
                                             proba_train=proba_train, proba_test=proba_test,
                                             beta=lr.coef_
                                             )


        grads_a.append(grad_a)
        grads_b.append(grad_b)

    return np.mean(grads_a), np.mean(grads_b), np.mean(roc_aucs), np.mean(losses)


def test_score_prediction_kernel_scaled(K, y, idx_train, idx_test, params):

    K_train, y_train = K[np.ix_(idx_train, idx_train)], y[np.ix_(idx_train)]
    K_test, y_test = K[np.ix_(idx_test, idx_train)], y[np.ix_(idx_test)]

    sc = StandardScaler(with_std=False)
    K_train = sc.fit_transform(K_train)
    K_test = sc.transform(K_test)

    clf = LogisticRegression(C=params['ml__C'], max_iter=10 ** 5)
    clf.fit(K_train, y_train)

    lr_best_score = roc_auc_score(y_test, clf.predict_proba(K_test).T[1])

    print "Test scores: ", lr_best_score

    return lr_best_score, MLE_l2_loss(y_test, clf.predict_log_proba(K_test), clf.coef_)


def test_score_prediction(K, y, idx_train, idx_test, params):

    K_train, y_train = K[np.ix_(idx_train, idx_train)], y[np.ix_(idx_train)]
    K_test, y_test = K[np.ix_(idx_test, idx_train)], y[np.ix_(idx_test)]

    # find best params on train

    # find best score on test
    clf = LogisticRegression(C=params['ml__C'], max_iter=10 ** 5)
    clf.fit(K_train, y_train)

    lr_best_score = roc_auc_score(y_test, clf.predict_proba(K_test).T[1])
    test_loss = MLE_l2_loss(y_test, clf.predict_log_proba(K_test), clf.coef_)
    print "Test scores: ", lr_best_score
    print "Test loss: ", test_loss

    return lr_best_score, test_loss



# def count_grads(K_train, y_train, da_train, db_train, params, dJ=None, scaled=False, kernel=False,
#                 with_template=False, n_splits=10, ndim=3, random_state=0):
#     if kernel:
#         exp_K_train = np.exp(-params['kernel__gamma'] * K_train)
#     else:
#         exp_K_train = K_train.copy()
#
#     roc_aucs = []
#     grads_a, grads_b = [], []
#     grads_J = []
#     hinge_losses = ][]

#     for idx_train, idx_test in StratifiedShuffleSplit(n_splits=n_splits, test_size=0.3,
#                                                       random_state=random_state).split(K_train, y_train):
#
#         exp_K_train_loc, exp_K_test_loc = exp_K_train[np.ix_(idx_train, idx_train)], exp_K_train[
#             np.ix_(idx_test, idx_train)]
#         y_train_loc, y_test_loc = y_train[idx_train], y_train[idx_test]
#
#         lr = LogisticRegression(n_jobs=1, max_iter=10 ** 5)
#         lr.set_params(**{'C': params['ml__C']})
#
#         lr.fit(exp_K_train_loc, y_train_loc)
#
#         roc_aucs.append(roc_auc_score(y_test_loc, lr.predict_proba(exp_K_test_loc).T[1]))
#
#         y_pred = lr.predict(exp_K_test_loc)
#         hinge_losses.append(np.mean(hinge_loss(y_pred, y_test_loc)))
#
#
#
#         if with_template:
#             if kernel:
#                 grad_a, grad_b, grad_J = diff_hinge_loss_lr(beta=lr.coef_,
#                                                             proba_test=lr.predict_proba(exp_K_test_loc).T[1],
#                                                             proba_train=lr.predict_proba(exp_K_train_loc).T[1],
#                                                             y_train=y_train_loc, y_test=y_test_loc,
#                                                             exp_K_test=exp_K_test_loc,
#                                                             exp_K_train=exp_K_train_loc,
#                                                             da_train=da_train[np.ix_(idx_train, idx_train)],
#                                                             db_train=db_train[np.ix_(idx_train, idx_train)],
#                                                             da_test=da_train[np.ix_(idx_test, idx_train)],
#                                                             db_test=db_train[np.ix_(idx_test, idx_train)],
#                                                             alpha=params['kernel__gamma'],
#                                                             dJ_train=dJ[np.ix_(idx_train, idx_train)],
#                                                             dJ_test=dJ[np.ix_(idx_test, idx_train)],
#                                                             scaled=scaled, kernel=kernel,
#                                                             with_template=with_template, ndim=ndim)
#             else:
#                 grad_a, grad_b, grad_J = diff_hinge_loss_lr(beta=lr.coef_,
#                                                             proba_test=lr.predict_proba(exp_K_test_loc).T[1],
#                                                             proba_train=lr.predict_proba(exp_K_train_loc).T[1],
#                                                             y_train=y_train_loc, y_test=y_test_loc,
#                                                             exp_K_test=exp_K_test_loc,
#                                                             exp_K_train=exp_K_train_loc,
#                                                             da_train=da_train[np.ix_(idx_train, idx_train)],
#                                                             db_train=db_train[np.ix_(idx_train, idx_train)],
#                                                             da_test=da_train[np.ix_(idx_test, idx_train)],
#                                                             db_test=db_train[np.ix_(idx_test, idx_train)],
#                                                             dJ_train=dJ[np.ix_(idx_train, idx_train)],
#                                                             dJ_test=dJ[np.ix_(idx_test, idx_train)],
#                                                             scaled=scaled, kernel=kernel,
#                                                             with_template=with_template, ndim=ndim)
#
#             grads_J.append(grad_J)
#         else:
#             if kernel:
#                 grad_a, grad_b = diff_hinge_loss_lr(beta=lr.coef_,
#                                                 proba_train=lr.predict_proba(exp_K_train_loc).T[1],
#                                                 proba_test=lr.predict_proba(exp_K_test_loc).T[1],
#                                                 y_train=y_train_loc, y_test=y_test_loc,
#                                                 exp_K_test=exp_K_test_loc,
#                                                 exp_K_train=exp_K_train_loc,
#                                                 da_train=da_train[np.ix_(idx_train, idx_train)],
#                                                 db_train=db_train[np.ix_(idx_train, idx_train)],
#                                                 da_test=da_train[np.ix_(idx_test, idx_train)],
#                                                 db_test=db_train[np.ix_(idx_test, idx_train)],
#                                                 alpha=params['kernel__gamma'],
#                                                 dJ_train=None, dJ_test=None,
#                                                 scaled=scaled, kernel=kernel,
#                                                 with_template=with_template, ndim=ndim)
#             else:
#                 grad_a, grad_b = diff_hinge_loss_lr(beta=lr.coef_,
#                                                     proba_train=lr.predict_proba(exp_K_train_loc).T[1],
#                                                     proba_test=lr.predict_proba(exp_K_test_loc).T[1],
#                                                     y_train=y_train_loc, y_test=y_test_loc,
#                                                     exp_K_test=exp_K_test_loc,
#                                                     exp_K_train=exp_K_train_loc,
#                                                     da_train=da_train[np.ix_(idx_train, idx_train)],
#                                                     db_train=db_train[np.ix_(idx_train, idx_train)],
#                                                     da_test=da_train[np.ix_(idx_test, idx_train)],
#                                                     db_test=db_train[np.ix_(idx_test, idx_train)],
#                                                     dJ_train=None, dJ_test=None,
#                                                     scaled=scaled, kernel=kernel,
#                                                     with_template=with_template, ndim=ndim)
#
#
#         grads_a.append(grad_a)
#         grads_b.append(grad_b)
#
#     print 'Hinge Loss Mean: ', np.mean(hinge_losses)
#     print 'ROC AUC MEAN: ', np.mean(roc_aucs)
#
#     if with_template:
#         return np.mean(grads_a), np.mean(grads_b), np.mean(grads_J, axis=0), np.mean(roc_aucs), np.mean(hinge_losses)
#
#     return np.mean(grads_a), np.mean(grads_b), np.mean(roc_aucs), np.mean(hinge_losses)
#

