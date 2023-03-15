"""Tidy up files.
"""
import glob
import os


def main():

    archive_dir = "figs/archive"
    words = ["figs/animation_*.gif", "figs/curve_*.png"]

    with open("run_info.txt", "r") as f:
        lines = f.readlines()
    lines = [line.split() for line in lines]
    variables = ["_".join(line[:2]) for line in lines]

    remain_files = [archive_dir]
    for word in words:
        for variable in variables:
            filepath = word.replace("*", variable)
            remain_files.append(filepath)

    target_files = glob.glob("figs/*")
    for f in target_files:
        basename = os.path.basename(f)
        dist_path = os.path.join(archive_dir, basename)
        if not (f in remain_files):
            os.rename(f, dist_path)
            print(f"{f} -> {dist_path}")


if __name__ == "__main__":
    main()
