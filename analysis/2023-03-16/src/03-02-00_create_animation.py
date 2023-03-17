import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def create_animation(savedir, extension='png'):
    file_list = sorted([os.path.join(savedir, f) for f
                        in os.listdir(savedir)
                        if f.endswith(extension)])
    fig, ax = plt.subplots()
    img = ax.imshow(plt.imread(file_list[0]))

    def update(frame):
        img.set_data(plt.imread(file_list[frame]))
        return img,

    ani = FuncAnimation(fig, update, frames=len(file_list),
                        interval=100, blit=True)
    return ani


def main():
    ani = create_animation('anim', extension='.png')
    ani.save('../figs/animation.gif', writer='pillow')
    plt.show()


if __name__ == "__main__":
    main()
