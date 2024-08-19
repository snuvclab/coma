import numpy as np


def mean_absolute_error(src, tgt, eps=1e-12):
    assert src.ndim == 1
    assert tgt.ndim == 1
    assert src.shape[0] == tgt.shape[0]

    src = src / (np.sum(src) + eps)
    tgt = tgt / (np.sum(tgt) + eps)

    return np.mean(np.absolute(src - tgt))


def mean_absolute_error_batch(src, tgt, eps=1e-12):
    assert src.ndim == 2
    assert tgt.ndim == 2
    assert src.shape == tgt.shape

    src = src / (np.sum(src, axis=-1, keepdims=True) + eps)  # B x N
    tgt = tgt / (np.sum(tgt, axis=-1, keepdims=True) + eps)  # B x N

    mae_batch = np.mean(np.absolute(src - tgt), axis=-1)  # B

    return np.mean(mae_batch)


def simlarity_metric(src, tgt, eps=1e-12):
    assert src.ndim == 1
    assert tgt.ndim == 1
    assert src.shape[0] == tgt.shape[0]

    src = src / (np.sum(src) + eps)
    tgt = tgt / (np.sum(tgt) + eps)

    return np.sum(np.min(np.stack([src, tgt], axis=-1), axis=-1))


def simlarity_metric_batch(src, tgt, eps=1e-12):
    assert src.ndim == 2
    assert tgt.ndim == 2
    assert src.shape == tgt.shape

    src = src / (np.sum(src, axis=-1, keepdims=True) + eps)  # B x N
    tgt = tgt / (np.sum(tgt, axis=-1, keepdims=True) + eps)  # B x N

    sim_batch = np.sum(np.min(np.stack([src, tgt], axis=-1), axis=-1), axis=-1)  # B

    return np.mean(sim_batch)


def quant_metrics_for_two_distributions(
    pred_dist,
    test_dist,
    eps,
):
    # # normalize both (again)
    pred_dist = pred_dist / (np.sum(pred_dist) + eps)
    test_dist = test_dist / (np.sum(test_dist) + eps)

    # MAE
    mae = mean_absolute_error(
        src=pred_dist,
        tgt=test_dist,
        eps=eps,
    )

    # SIM
    sim = simlarity_metric(
        src=pred_dist,
        tgt=test_dist,
        eps=eps,
    )

    return {
        "mae": mae.item(),
        "sim": sim.item(),
    }


def quant_metrics_for_two_batch_of_distributions(
    batch_pred_dist,  # B x N
    batch_test_dist,  # B x N
    eps,
):
    # # normalize both (again)
    batch_pred_dist = batch_pred_dist / (np.sum(batch_pred_dist, axis=-1, keepdims=True) + eps)
    batch_test_dist = batch_test_dist / (np.sum(batch_test_dist, axis=-1, keepdims=True) + eps)

    # MAE
    mae = mean_absolute_error(
        src=batch_pred_dist,
        tgt=batch_test_dist,
        eps=eps,
    )

    # SIM
    sim = simlarity_metric_batch(
        src=batch_pred_dist,
        tgt=batch_test_dist,
        eps=eps,
    )

    return {
        "mae": mae.item(),
        "sim": sim.item(),
    }
