import numpy as np

def vec_rejection_val(
    init_states,          # shape (N,)
    proposal_rvs,         # fn size=(K,P), src→ draws of shape (K,P)
    val,                  # fn X, S → log‑weight array, shape (K,P)
    B,                    # either scalar or fn S→ array shape (K,)
    batch_size=None
):
    """
    For each starting state s in init_states, draw one sample x ~ q(·|s) by rejection:
      accept x when u < exp(val(x,s) - B(s)).
    Returns array of accepted x's, one per init state.
    """
    N = init_states.shape[0]
    d = init_states.shape[1]
    out  = np.empty((N, d))
    done = np.zeros(N, bool)
    total_proposals = 0 


    # if no batch_size, pick something so you get ~1 acceptance per iteration
    if batch_size is None:
        # a rough α ≈ mean[exp(val(s_sample,s) - B(s))]
        # we just draw one proposal per state to estimate α
        test_x = proposal_rvs(size=N, src=init_states)
        # print("shape of test_x", test_x.shape)
        # print("shape of vals", np.exp(val(test_x)).shape)
        est_R  = np.exp(val(test_x))/np.exp(B)  # shape (N,)
        # print("shape of est_R", est_R.shape)
        alpha  = np.mean(est_R)
        batch_size = max(1, int(np.ceil(1/alpha)))
    # print("batch_size", batch_size)
    

    while not done.all():
        active = np.where(~done)[0]       # indices still needing a sample
        K      = active.size

        # 1) draw K×batch_size proposals in one go
        X_prop = proposal_rvs(size=batch_size, src=init_states[active])
        # print("X_prop", X_prop.shape)
        total_proposals += K * batch_size 
        U      = np.random.rand(batch_size, K)

        # 2) compute bound and log‑weights
        Bvals = B          # shape (K,) or scalar
        V = val(X_prop)  # shape (K, batch_size)
        # 3) acceptance ratios
        R = np.exp(V)/(np.ones_like(V)*np.exp(B))                          # ≤1 by construction
        # print("Val(X_prop)/B", R.shape)
        accept = (U < R)                              # boolean mask
        # print("accept", accept.shape)
        # print("active", active.shape)
        # 4) for each chain, find first accepted proposal
        got_any = accept.any(axis=0)                  # which chains accepted
        # print("got_any", got_any.shape)
        if got_any.any():
            first_idx = np.argmax(accept, axis=0)     # first True per chain
            sel       = active[got_any]               # original indices
            picks     = first_idx[got_any]            # column of first accept
            out[sel]  = X_prop[picks, got_any, :]
            done[sel] = True

    return out, total_proposals
def double_rejection(
    init_states,          # shape (N,)
    proposal_rvs,         # fn size=(K,P), src→ draws of shape (K,P)
    v1,                  # fn X, S → log‑weight array, shape (K,P)
    v2, # array of shape (N, ) of current values
    batch_size=None
):
    """
    For each starting state s in init_states, draw one sample x ~ q(·|s) by rejection:
      accept x when u < exp(val(x,s) - B(s)).
    Returns array of accepted x's, one per init state.
    """
    N = init_states.shape[0]
    d = init_states.shape[1]
    out  = np.empty((N, d))
    done = np.zeros(N, bool)
    total_proposals = 0 


    # if no batch_size, pick something so you get ~1 acceptance per iteration
    if batch_size is None:
        # a rough α ≈ mean[exp(val(s_sample,s) - B(s))]
        # we just draw one proposal per state to estimate α
        test_x = proposal_rvs(size=N, src=init_states)
        # print("shape of test_x", test_x.shape)
        # print("shape of vals", np.exp(val(test_x)).shape)
        est_R  = np.exp(v1(test_x) - v2)  # shape (N,)
        # print("shape of est_R", est_R.shape)
        alpha  = np.mean(est_R)
        batch_size = max(1, int(np.ceil(1/alpha)))
    # print("batch_size", batch_size)
    

    while not done.all():
        active = np.where(~done)[0]       # indices still needing a sample
        K      = active.size

        # 1) draw K×batch_size proposals in one go
        X_prop = proposal_rvs(size=batch_size, src=init_states[active])
        # print("X_prop", X_prop.shape)
        total_proposals += K * batch_size 
        U      = np.random.rand(batch_size, K)

        # 2) compute bound and log‑weights
        V1 = v1(X_prop)  # shape (K, batch_size)
        # 3) acceptance ratios
        R = np.exp(V1 - v2[active])                  # ≤1 by construction
        # print("Val(X_prop)/B", R.shape)
        accept = (U < R)                              # boolean mask
        # print("accept", accept.shape)
        # print("active", active.shape)
        # 4) for each chain, find first accepted proposal
        got_any = accept.any(axis=0)                  # which chains accepted
        # print("got_any", got_any.shape)
        if got_any.any():
            first_idx = np.argmax(accept, axis=0)     # first True per chain
            sel       = active[got_any]               # original indices
            picks     = first_idx[got_any]            # column of first accept
            out[sel]  = X_prop[picks, got_any, :]
            done[sel] = True

    return out, total_proposals
