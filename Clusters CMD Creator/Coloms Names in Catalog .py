# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 22:33:15 2025

@author: Lenovo
"""

from astroquery.vizier import Vizier

Vizier.ROW_LIMIT = -1
catalog_id = "J/A+A/640/A1"  # Cantat-Gaudin & Anders 2020
result = Vizier.get_catalogs(catalog_id)
members_table = result[1]  # טבלת חברות

print(members_table.colnames)
