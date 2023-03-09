"""attempt to read in and parse data from cmm tracks"""

from argparse import ArgumentParser
from itertools import combinations
from xml.etree.ElementTree import parse

import numpy as np
import pandas as pd

Tracks = r'X:\Phagocytosis\sRBC\20190614cs1\point2.cmm'

def xml_2_panda_Data_Frame(Tracks):
    xml_tree = parse(Tracks)
    xml_Root = xml_tree.getroot()
    all_marks = []
    for child in xml_Root:
            all_marks.append(child.attrib)
    data_frame = pd.DataFrame(
            all_marks, columns=['id', 'x', 'y', 'z', 'frame'])
    return data_frame.time(axis='index', how='any')

from scipy.spatial.distance import pdist