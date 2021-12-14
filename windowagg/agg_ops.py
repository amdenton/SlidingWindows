"""
Last updated on Tue Dec 14

@authors: Anne Denton, David Schwarz, Rahul Gomes

License information:
https://opensource.org/licenses/GPL-3.0
"""
from enum import Enum

class Agg_ops(Enum):
    add_all = 1
    add_bottom = 2
    add_right = 3
    add_main_diag = 4
    maximum = 5
    minimum = 6