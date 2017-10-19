import os
import shutil
import subprocess
import sys

import tensorflow as tf


def start_tensorboard(directory):
    tensorboard_path = sys.executable.replace("python.exe", "") + "Scripts\\tensorboard.exe"

    command = "start http://localhost:6006/#graphs & " + sys.executable + " " + tensorboard_path + " --logdir=" + directory

    # TODO safely quit command
    s = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)

    try:
        s.wait(5)
    except:
        pass
    s.terminate()

def open_tensorboard(file_dir, tensorflow_session):
    if os.path.exists(file_dir):
        directory = "tensorboard/" + os.path.basename(file_dir).replace(".py", "")

        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            shutil.rmtree(directory)
            os.mkdir(directory)

        tf.summary.FileWriter(directory, tensorflow_session.graph)

        start_tensorboard(directory)

        return directory
    return None
