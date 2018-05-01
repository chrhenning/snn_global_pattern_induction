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
@title           :observer/observable.py
@author          :ch
@contact         :christian@ini.ethz.ch
@created         :04/10/2017
@version         :1.0
@python_version  :3.5.2

An class representing the Observable object in the Observer Pattern.
"""

import configuration as config

import logging
logger = logging.getLogger(config.logging_name)

class Observable(object):
    """The class code has been copied from:
    http://www.giantflyingsaucer.com/blog/?p=5117

    The copyright holder is Chad Lung. The code was published on
    October 7, 2014 (Accessed: April 10, 2017).

    Originally, the code has been adopted from Wikipedia. Therefore, it is
    assumed that the code is licensed under CC BY-SA:
    https://creativecommons.org/licenses/by-sa/3.0/
    """

    def __init__(self):
        self.observers = []

    def register(self, observer):
        if not observer in self.observers:
            self.observers.append(observer)

    def unregister(self, observer):
        if observer in self.observers:
            self.observers.remove(observer)

    def unregister_all(self):
        if self.observers:
            del self.observers[:]

    def update_observers(self, *args, **kwargs):
        for observer in self.observers:
            observer.update(*args, **kwargs)

if __name__ == '__main__':
    pass


