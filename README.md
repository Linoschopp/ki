# ki

vlguser@vlg-R101-21:~$ pyenv install 3.13
Downloading Python-3.13.0.tar.xz...
-> https://www.python.org/ftp/python/3.13.0/Python-3.13.0.tar.xz
Installing Python-3.13.0...
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import bz2
  File "/home/vlguser/.pyenv/versions/3.13.0/lib/python3.13/bz2.py", line 17, in <module>
    from _bz2 import BZ2Compressor, BZ2Decompressor
ModuleNotFoundError: No module named '_bz2'
WARNING: The Python bz2 extension was not compiled. Missing the bzip2 lib?
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import curses
  File "/home/vlguser/.pyenv/versions/3.13.0/lib/python3.13/curses/__init__.py", line 13, in <module>
    from _curses import *
ModuleNotFoundError: No module named '_curses'
WARNING: The Python curses extension was not compiled. Missing the ncurses lib?
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import ctypes
  File "/home/vlguser/.pyenv/versions/3.13.0/lib/python3.13/ctypes/__init__.py", line 8, in <module>
    from _ctypes import Union, Structure, Array
ModuleNotFoundError: No module named '_ctypes'
WARNING: The Python ctypes extension was not compiled. Missing the libffi lib?
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import readline
ModuleNotFoundError: No module named 'readline'
WARNING: The Python readline extension was not compiled. Missing the GNU readline lib?
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import sqlite3
  File "/home/vlguser/.pyenv/versions/3.13.0/lib/python3.13/sqlite3/__init__.py", line 57, in <module>
    from sqlite3.dbapi2 import *
  File "/home/vlguser/.pyenv/versions/3.13.0/lib/python3.13/sqlite3/dbapi2.py", line 27, in <module>
    from _sqlite3 import *
ModuleNotFoundError: No module named '_sqlite3'
WARNING: The Python sqlite3 extension was not compiled. Missing the SQLite3 lib?
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import tkinter
  File "/home/vlguser/.pyenv/versions/3.13.0/lib/python3.13/tkinter/__init__.py", line 38, in <module>
    import _tkinter # If this fails your Python may not be configured for Tk
    ^^^^^^^^^^^^^^^
ModuleNotFoundError: No module named '_tkinter'
WARNING: The Python tkinter extension was not compiled and GUI subsystem has been detected. Missing the Tk toolkit?
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import lzma
  File "/home/vlguser/.pyenv/versions/3.13.0/lib/python3.13/lzma.py", line 27, in <module>
    from _lzma import *
ModuleNotFoundError: No module named '_lzma'
WARNING: The Python lzma extension was not compiled. Missing the lzma lib?
Installed Python-3.13.0 to /home/vlguser/.pyenv/versions/3.13.0
vlguser@vlg-R101-21:~$ 
