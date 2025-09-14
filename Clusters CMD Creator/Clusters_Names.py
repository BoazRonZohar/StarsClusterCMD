# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 22:16:55 2025

@author: Lenovo
"""

from astroquery.vizier import Vizier
Vizier.ROW_LIMIT = -1
cat = Vizier.get_catalogs("J/A+A/640/A1")
members_table = cat[1]
print(set(members_table["Cluster"]))
