"""Tidy up files.
"""
import glob
import os


def archive_files(target_path, archive_dir, remain_files, verbose=False):
    target_files = [p for p in glob.glob(target_path)
                    if os.path.isfile(p)]

    for f in target_files:
        basename = os.path.basename(f)
        dist_path = os.path.join(archive_dir, basename)
        if not (f in remain_files):
            os.rename(f, dist_path)
            if verbose:
                print(f"{f} -> {dist_path}")


def create_ramain_list(runinfo_path, basenames):
    with open(runinfo_path, "r") as f:
        lines = f.readlines()
    lines = [line.split() for line in lines]
    variables = ["_".join(line[:2]) for line in lines]

    remain_files = []
    for basename in basenames:
        for variable in variables:
            filepath = basename.replace("*", variable)
            remain_files.append(filepath)

    return remain_files


def main():

    basenames = ["../figs/animation_*.gif", "../figs/curve_*.png"]
    remain_files_auto = create_ramain_list("run_info.txt", basenames)
    remain_files_manual = ["../figs/animation.gif"]
    remain_files = remain_files_auto + remain_files_manual

    target_path = "../figs/*"
    archive_dir = "../figs/archive"
    archive_files(target_path, archive_dir, remain_files, verbose=True)


if __name__ == "__main__":
    main()
