import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure
from matplotlib.axes import Axes

def plot_u(us: list[np.array], show_labels = False, timeout=2000):
    """Plot the u values (i.e., temperature of the rooms)
    Written with some help by copilot"""
    fig = plt.figure(figsize=(9, 6))

    room_width = int(np.sqrt(us[0].size))
    room_height = int(us[0].size/room_width)
    vmin = min(u.min() for u in us)
    vmax = max(u.max() for u in us)

    gs = GridSpec(room_height*2-1, 3,
                  figure=fig,
                #   width_ratios=[1, 1, 1],
                #   height_ratios=[1]*(room_height*2-1),
                  wspace=0.0, # no gaps
                  hspace=0.0)

    ax1 = fig.add_subplot(gs[room_height-1:, 0])
    ax2 = fig.add_subplot(gs[:, 1])
    ax3 = fig.add_subplot(gs[:room_height, 2])

    for ax, u in zip((ax1, ax2, ax3), us):
        u = u.reshape(-1,room_width)
        im = ax.imshow(u,
                       vmin=vmin,
                       vmax=vmax,
                       cmap = "hot",
                       aspect="auto") # , cmap="viridis"
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)
        # Add labels
        if show_labels:
            for i in range(u.shape[0]):
                for j in range(u.shape[1]):
                    ax.text(
                        j, i, # x, y position
                        f"{u[i, j]:.1f}", # label
                        ha="center",
                        va="center",
                        color="white"
                    )

    plt.tight_layout()
    fig.subplots_adjust(left=0, right=1, bottom=0.05, top=0.95)
    # Shared colorbar below all rooms
    cbar = fig.colorbar(
        im,
        ax=[ax1, ax2, ax3],
        orientation="horizontal",
        pad=0.01,
        shrink=0.8
        )
    cbar.set_label("Value")

    if timeout is not None:
        timer = fig.canvas.new_timer(interval=timeout) # ms
        timer.add_callback(lambda: plt.close(fig))
        timer.start()

    plt.show()

