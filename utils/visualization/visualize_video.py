import os
from glob import glob
import shutil
import cv2


def generate_video_from_imgs(images_save_directory, fps=15.0, delete_dir=True):
    # delete videos if exists
    if os.path.exists(f"{images_save_directory}.mp4"):
        os.remove(f"{images_save_directory}.mp4")
    if os.path.exists(f"{images_save_directory}_before_process.mp4"):
        os.remove(f"{images_save_directory}_before_process.mp4")

    # assume there are "enumerated" images under "images_save_directory"
    assert os.path.isdir(images_save_directory)
    ImgPaths = sorted(list(glob(f"{images_save_directory}/*")))

    if len(ImgPaths) == 0:
        print("\tSkipping, since there must be at least one image to create mp4\n")
    else:
        # mp4 configuration
        video_path = images_save_directory + "_before_process.mp4"

        # Get height and width config
        images = sorted([ImgPath.split("/")[-1] for ImgPath in ImgPaths if ImgPath.endswith(".png")])
        frame = cv2.imread(os.path.join(images_save_directory, images[0]))
        height, width, channels = frame.shape

        # create mp4 video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        for image in images:
            video.write(cv2.imread(os.path.join(images_save_directory, image)))
        cv2.destroyAllWindows()
        video.release()

        # generated video is not compatible with HTML5. Post-process and change codec of video, so that it is applicable to HTML.
        os.system(f'ffmpeg -i "{images_save_directory}_before_process.mp4" -vcodec libx264 -f mp4 "{images_save_directory}.mp4" -loglevel "quiet"')

    # remove group of images, and remove video before post-process.
    if delete_dir and os.path.exists(images_save_directory):
        shutil.rmtree(images_save_directory)
    # remove 'before-process' video
    if os.path.exists(f"{images_save_directory}_before_process.mp4"):
        os.remove(f"{images_save_directory}_before_process.mp4")
