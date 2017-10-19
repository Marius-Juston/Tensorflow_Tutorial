import os
import sys

import tensorflow as tf


def open_tensorboard(file_dir, tensorflow_session):
    if os.path.exists(file_dir):
        directory = "tensorboard/" + os.path.basename(file_dir).replace(".py", "")

        file_writer = tf.summary.FileWriter(directory, tensorflow_session.graph)

        tensorboard_path = sys.executable.replace("python.exe", "") + "Scripts\\tensorboard.exe"
        command = "start cmd /C " + tensorboard_path + " --logdir=" + directory

        os.system(command)
        os.system("start http://localhost:6006/#graphs")

        return file_writer, directory
    return None
