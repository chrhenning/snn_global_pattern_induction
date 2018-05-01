#!/usr/bin/env python3
# Copyright 2017 Christian Henning
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
@title           :pypatterns/singleton.py
@author          :ch
@contact         :christian@ini.ethz.ch
@created         :04/18/2017
@version         :1.0
@python_version  :3.5.2

Implementation that alows the usage of the Singleton pattern in Python.
"""

class Singleton(type):
    """The class code has been copied from:
    http://www.python-course.eu/python3_metaclasses.php

    The copyright holder is Bernd Klein. (Accessed: April 18, 2017).

    assumed that the code is licensed under CC BY 3.0:
    https://creativecommons.org/licenses/by/3.0/
    """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args,
                                                                 **kwargs)
        return cls._instances[cls]

if __name__ == '__main__':
    pass


