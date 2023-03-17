"""Create animation from snapshot.
"""
from PIL import Image
import glob


def main():
    image_files = sorted(glob.glob("../figs/anim/animation_???.png"))
    images = [Image.open(f) for f in image_files]
    images[0].save("../figs/animation.gif", save_all=True,
                   append_images=images[1:], duration=100,
                   loop=1)


if __name__ == "__main__":
    main()
