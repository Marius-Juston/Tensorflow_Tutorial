import os

import tensorflow as tf


def open_tensorboard(file_dir, tensorflow_session):
    if os.path.exists(file_dir):
        directory = "tensorboard/" + os.path.basename(file_dir).replace(".py", "")

        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            import shutil

            shutil.rmtree(directory)
            os.mkdir(directory)

        tf.summary.FileWriter(directory, tensorflow_session.graph)

        import sys

        python_path = sys.executable
        tensorboard_path = sys.executable.replace("python.exe", "") + "Scripts\\tensorboard.exe"

        os.system(
            "start http://localhost:6006/#graphs & "
            + python_path + " " + tensorboard_path + " --logdir=" + directory
        )
