import os
import shutil
import sys

import psutil
import tensorflow as tf
import win32gui
import win32process


def open_tensorboard(file_dir, tensorflow_session=None, if_exists_clean=True):
    if os.path.exists(file_dir):
        program_name = os.path.basename(file_dir).replace(".py", "")

        directory = "tensorboard/" + program_name

        tensorboard_title_prefix = 'Tensorflow - '

        kill_all_already_running_tensorboard(tensorboard_title_prefix)

        if if_exists_clean and os.path.exists(directory):
            shutil.rmtree(directory)

        file_writer = tf.summary.FileWriter(directory)

        if tensorflow_session:
            file_writer.add_graph(tensorflow_session.graph)

        cmd_title = tensorboard_title_prefix + program_name

        tensorboard_path = sys.executable.replace("python.exe", "") + "Scripts\\tensorboard.exe"
        command = 'start "' + cmd_title + '" cmd /C ' + tensorboard_path + " --logdir=" + directory

        print(command)

        os.system(command)
        os.system("start http://localhost:6006/#graphs")

        return file_writer, directory
    return None


kill_command = "taskkill /PID "

result = None


def enumWindowsProc(hwnd, lParam):
    global search
    global result

    if (lParam is None) or ((lParam is not None) and (win32process.GetWindowThreadProcessId(hwnd)[1] == lParam)):
        text = win32gui.GetWindowText(hwnd)

        print(text)

        if search in text:
            result = True


def enumProcWnds(pid=None):
    win32gui.EnumWindows(enumWindowsProc, pid)


search = None


def kill_all_already_running_tensorboard(search_title):
    global search
    global result
    global kill_command
    search = search_title

    for process in psutil.process_iter():
        try:
            if process.name() == u"cmd.exe":
                enumProcWnds(process.pid)

                if result:
                    result = False
                    os.system(kill_command + str(process.pid))
        except psutil.AccessDenied:
            print("Permission error or access denied on process")

    search = None
