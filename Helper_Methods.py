import os
import shutil
import sys

import tensorflow as tf


def open_tensorboard(file_dir, tensorflow_session):
    if os.path.exists(file_dir):
        directory = "tensorboard/" + os.path.basename(file_dir).replace(".py", "")

        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            shutil.rmtree(directory)
            os.mkdir(directory)

        tf.summary.FileWriter(directory, tensorflow_session.graph)

        tensorboard_path = sys.executable.replace("python.exe", "") + "Scripts\\tensorboard.exe"
        command = "start cmd /C " + tensorboard_path + " --logdir=" + directory

        os.system(command)
        os.system("start http://localhost:6006/#graphs")

        return directory
    return None
