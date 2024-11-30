import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display, HTML
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def fig_to_image(fig: plt.Figure):
    canvas = FigureCanvas(figure=fig)
    canvas.draw()

    fig.tight_layout(pad=2.0)


    width, height = fig.get_size_inches() * fig.get_dpi()
    rgb_array = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    rgb_array = rgb_array.reshape(int(height), int(width), 4)

    plt.close(fig)

    return rgb_array


def display_frames_as_gif(frames):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1)

    patch = ax.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(
        fig, animate, frames=len(frames), interval=1000
    )
    video = anim.to_html5_video()
    html = HTML(video)
    display(html)
    plt.close()