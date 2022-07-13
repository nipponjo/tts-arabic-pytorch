#import matplotlib
# matplotlib.use("Agg")
import matplotlib.pylab as plt


def get_alignment_figure(img):
    fig = plt.figure(figsize=(6, 4))
    plt.imshow(img, aspect='auto', origin='lower',
               interpolation='none')
    plt.xlabel('Spectrogram frame')
    plt.ylabel('Input token')
    plt.colorbar()
    plt.tight_layout()
    return fig


def get_spectrogram_figure(spec):
    fig = plt.figure(figsize=(12, 3))
    plt.imshow(spec, aspect='auto', origin='lower',
               interpolation='none')
    plt.xlabel('Frame')
    plt.ylabel('Channel')
    plt.colorbar()
    plt.tight_layout()
    return fig


def get_specs_figure(specs, xlabels):
    n = len(specs)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3*n))

    for i, ax in enumerate(axes):
        im = ax.imshow(specs[i], aspect='auto', origin='lower',
                       interpolation='none')
        ax.set_xlabel(xlabels[i])
        ax.set_ylabel('Channel')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    return fig
