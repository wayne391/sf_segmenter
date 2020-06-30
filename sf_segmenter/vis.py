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
        data,
        boundaries=None,
        outdir=None,
        vis_bounds=False):

    if outdir:
        os.makedirs(outdir, exist_ok=True)
    print(' [o] save to ...', outdir)
        
    # plot input feature
    if data['input'] is not None:
        plt.figure()
        plt.imshow(data['input'].T, interpolation="nearest", aspect="auto")
        plt.title('input feature')
        plt.savefig(os.path.join(outdir, 'input.png'))
        # plt.show()
        plt.close()

    # plot recurrence plot (R)
    if data['R'] is not None:
        plt.figure(figsize=(5, 5))
        plt.imshow(data['R'], interpolation="nearest", cmap=plt.get_cmap("binary"))
        if vis_bounds and boundaries:
            [plt.axvline(p, color="red", linestyle=':') for p in boundaries]
            [plt.axhline(p, color="red", linestyle=':') for p in boundaries]
        plt.title('recurrence plot')
        plt.savefig(os.path.join(outdir, 'R.png'))
        # plt.show()
        plt.close()

    # plot time-lag (L)
    if data['L'] is not None:
        plt.figure(figsize=(5, 5))
        plt.imshow(data['L'], interpolation="nearest", cmap=plt.get_cmap("binary"))
        if vis_bounds and boundaries:
            [plt.axvline(p, color="red", linestyle=':') for p in boundaries]
            [plt.axhline(p, color="red", linestyle=':') for p in boundaries]
        plt.savefig(os.path.join(outdir, 'L.png'))
        # plt.show()
        plt.close()

    #  plot smoothed time-lag (SF)
    if data['SF'] is not None:
        plt.figure(figsize=(5, 5))
        plt.imshow(data['SF'].T, interpolation="nearest", cmap=plt.get_cmap("binary"))
        plt.title('SF (after filtering)')
        if vis_bounds and boundaries:
            [plt.axvline(p, color="red", linestyle=':') for p in boundaries]
            [plt.axhline(p, color="red", linestyle=':') for p in boundaries]
        plt.savefig(os.path.join(outdir, 'SF.png'))
        # plt.show(block=False))
        plt.close()

    # plot novelty cureve (nc)
    if data['nc'] is not None:
        plt.figure()
        plt.plot(data['nc'])
        if boundaries is not None:
            [plt.axvline(p, color="green", linestyle=':') for p in boundaries]
        plt.title('novelty curve')
        plt.savefig(os.path.join(outdir, 'nc.png'))
        # plt.show()
        plt.close()
