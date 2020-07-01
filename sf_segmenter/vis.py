import os
import matplotlib.pyplot as plt

"""
    data = {
        'input': segmenter.F,
        'R': segmenter.R,
        'L': segmenter.L,
        'SF': segmenter.SF,
        'nc': segmenter.nc,
    }

"""
def plot_feats(
        F=None,
        R=None,
        L=None,
        SF=None,
        nc=None,
        S=None,
        S_trans=None,
        S_final=None,
        boundaries=None,
        outdir=None,
        vis_bounds=False):

    if outdir:
        os.makedirs(outdir, exist_ok=True)
    print(' [o] save to ...', outdir)
        
    # plot input feature
    if F is not None:
        plt.figure()
        plt.imshow(F.T, interpolation="nearest", aspect="auto")
        plt.title('input feature')
        plt.savefig(os.path.join(outdir, 'input.png'))
        # plt.show()

    # plot recurrence plot (R)
    if R is not None:
        plt.figure(figsize=(5, 5))
        plt.imshow(R, interpolation="nearest", cmap=plt.get_cmap("binary"))
        if vis_bounds and boundaries is not None:
            [plt.axvline(p, color="red", linestyle=':') for p in boundaries]
            [plt.axhline(p, color="red", linestyle=':') for p in boundaries]
        plt.title('recurrence plot')
        plt.savefig(os.path.join(outdir, 'R.png'))
        # plt.show()

    # plot time-lag (L)
    if L is not None:
        plt.figure(figsize=(5, 5))
        plt.imshow(L, interpolation="nearest", cmap=plt.get_cmap("binary"))
        if vis_bounds and boundaries is not None:
            [plt.axvline(p, color="red", linestyle=':') for p in boundaries]
            [plt.axhline(p, color="red", linestyle=':') for p in boundaries]
        plt.savefig(os.path.join(outdir, 'L.png'))
        # plt.show()

    #  plot smoothed time-lag (SF)
    if SF is not None:
        plt.figure(figsize=(5, 5))
        plt.imshow(SF.T, interpolation="nearest", cmap=plt.get_cmap("binary"))
        plt.title('SF (after filtering)')
        if vis_bounds and boundaries is not None:
            [plt.axvline(p, color="red", linestyle=':') for p in boundaries]
            [plt.axhline(p, color="red", linestyle=':') for p in boundaries]
        plt.savefig(os.path.join(outdir, 'SF.png'))
        # plt.show(block=False))

    # plot novelty cureve (nc)
    if nc is not None:
        plt.figure()
        plt.plot(nc)
        if boundaries is not None:
            [plt.axvline(p, color="green", linestyle=':') for p in boundaries]
        plt.title('novelty curve')
        plt.savefig(os.path.join(outdir, 'nc.png'))
        # plt.show()

    # labeling features
    if S is not None:
        plt.figure(figsize=(5, 5))
        plt.imshow(S, interpolation="nearest", cmap=plt.get_cmap("binary"))
        plt.savefig(os.path.join(outdir, 'lab_S.png'))
        plt.title('labeling: S')

    if S_trans is not None:
        plt.figure(figsize=(5, 5))
        plt.imshow(S_trans, interpolation="nearest", cmap=plt.get_cmap("binary"))
        plt.savefig(os.path.join(outdir, 'lab_S_trans.png'))
        plt.title('labeling: S trans')

    if S_final is not None:
        plt.figure(figsize=(5, 5))
        plt.imshow(S_final, interpolation="nearest", cmap=plt.get_cmap("binary"))
        plt.savefig(os.path.join(outdir, 'lab_S_final.png'))
        plt.title('labeling: S final')
