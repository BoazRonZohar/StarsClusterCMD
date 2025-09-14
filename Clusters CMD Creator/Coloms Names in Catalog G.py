# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 21:14:46 2025

@author: Lenovo
"""

from astroquery.vizier import Vizier
catalog_id = "J/MNRAS/505/5978"
Vizier.ROW_LIMIT = 1  # מספיק לנו שורה אחת
result = Vizier.get_catalogs(catalog_id)
members_table = result[0]
print(members_table.colnames)
