"""Building the training set: trial splits, retrieval pool, pairing.

Order matters here, and getting it wrong is the easiest way to leak information:

  1. the imagery trials are split **before** any replication, within each stimulus,
     so that no copy of a test trial can appear in training;
  2. the retrieved perception trials are split independently, keeping the three
     repetitions of the same NSD image together, so that no repetition of a
     validation image is used for training;
  3. only then are the few imagery training trials replicated, with a small amount
     of noise on the copies, until they match the number of perceptual targets.

The held-out imagery trials of a stimulus are averaged into a single test input, to
raise its signal-to-noise ratio: one reconstruction per stimulus.
"""
import numpy as np
import torch

import settings as C


def split_trials(stim_ids, stimuli, rng, test_frac=C.TEST_FRAC, val_frac=C.VAL_FRAC):
    """Per-stimulus split of the imagery trials into train / validation / test."""
    train, val, test = {}, {}, {}
    for s in stimuli:
        idx = rng.permutation(np.where(stim_ids == s)[0])
        n_test = max(2, int(round(len(idx) * test_frac)))
        test[s] = idx[:n_test]
        rest = idx[n_test:]
        n_val = max(1, int(round(len(rest) * val_frac)))
        val[s], train[s] = rest[:n_val], rest[n_val:]
    sets = [set(np.concatenate([d[s] for s in stimuli]).tolist()) for d in (train, val, test)]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), "split overlaps"
    return train, val, test


def repetition_groups(rows, g=3):
    """Split a block of rows into groups of `g` consecutive repetitions.

    The phase is local to the block: repetitions of one NSD image are consecutive
    inside a block, and the block boundary resets the grouping. A leftover shorter
    than `g` stays together as a partial group.
    """
    start, n = rows[0], len(rows)
    groups = [np.arange(start + g * j, start + g * j + g) for j in range(n // g)]
    if n % g:
        groups.append(np.arange(start + n - (n % g), start + n))
    return groups


def replicate_with_noise(base, n_target, rng, noise=C.NOISE):
    """Repeat `base` up to `n_target` rows; perturb only the copies.

    The originals are left untouched, so the real trials stay in the training set
    exactly as measured. The noise is scaled by the standard deviation of the trials
    of that stimulus, and is small compared with the dispersion between real trials:
    its only purpose is to avoid exact duplicates.
    """
    n = len(base)
    idx = np.resize(np.arange(n), n_target)
    out = base[idx].copy()
    if n_target > n:
        out[n:] += noise * base.std() * rng.standard_normal(out[n:].shape).astype(out.dtype)
    return out


def build_dataset(imagery_fp, augmentation_fp, stim_set, seed=C.SEED):
    """Assemble training, validation and test tensors for one subject and set.

    Returns a dict with X_train/Y_train (imagery input, perceptual target), the same
    for validation, integer stimulus labels for the contrastive term, and X_test with
    one averaged input per stimulus.
    """
    rng = np.random.default_rng(seed)
    stimuli = np.array(C.BLOCK_ORDER[stim_set])

    d = torch.load(imagery_fp, map_location="cpu", weights_only=False)
    X, stim_ids = d["brain_tensor"].numpy(), np.array(d["stim_ids"])
    assert set(stimuli) == set(stim_ids), "stimulus names do not match the block order"
    tr_idx, va_idx, te_idx = split_trials(stim_ids, stimuli, rng)

    Y = torch.load(augmentation_fp, map_location="cpu", weights_only=False)
    Y = Y.numpy() if hasattr(Y, "numpy") else np.asarray(Y)
    rows_per_block = Y.shape[0] // len(stimuli)

    # --- split the retrieval pool by groups of three repetitions ------------------
    pool = {}
    for b, s in enumerate(stimuli):
        groups = repetition_groups(np.arange(b * rows_per_block, (b + 1) * rows_per_block))
        order = rng.permutation(len(groups))
        cut = int(round(len(order) * 0.90))
        g_train = [groups[k] for k in order[:cut]]
        g_val = [groups[k] for k in order[cut:]]
        pool[s] = (np.concatenate(g_train), np.concatenate(g_val))

    # every stimulus contributes the same number of targets, rounded down to a
    # multiple of three so that no group is cut in half and nothing is resampled
    n_train = (min(len(v[0]) for v in pool.values()) // 3) * 3

    X_tr, Y_tr, X_va, Y_va, L_tr, L_va = [], [], [], [], [], []
    for j, s in enumerate(stimuli):
        rows_tr, rows_va = pool[s]
        Y_tr.append(Y[rng.choice(rows_tr, n_train, replace=False)])
        Y_va.append(Y[rng.choice(rows_va, C.N_VAL, replace=C.N_VAL > len(rows_va))])
        X_tr.append(replicate_with_noise(X[tr_idx[s]], n_train, rng))
        X_va.append(replicate_with_noise(X[va_idx[s]], C.N_VAL, rng))
        L_tr.append(np.full(n_train, j))
        L_va.append(np.full(C.N_VAL, j))

    X_test = np.stack([X[te_idx[s]].mean(0) for s in stimuli])   # one input per stimulus

    print(f"  trials per stimulus: train {len(tr_idx[stimuli[0]])}, "
          f"val {len(va_idx[stimuli[0]])}, test {len(te_idx[stimuli[0]])} (averaged)")
    print(f"  targets per stimulus: train {n_train}, val {C.N_VAL}")
    return dict(
        X_train=np.concatenate(X_tr), Y_train=np.concatenate(Y_tr),
        X_val=np.concatenate(X_va), Y_val=np.concatenate(Y_va),
        lab_train=np.concatenate(L_tr), lab_val=np.concatenate(L_va),
        X_test=X_test, stimuli=stimuli,
    )
