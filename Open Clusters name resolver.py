# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 00:25:52 2025

@author: Lenovo
"""

from astroquery.vizier import Vizier

# Load catalog of open clusters
Vizier.ROW_LIMIT = -1
cat = Vizier.get_catalogs("J/A+A/640/A1")
members_table = cat[1]
catalog_clusters = set(members_table["Cluster"])

# Dictionary mapping Messier designations to catalog names
messier_to_catalog = {
    "M6":   "NGC_6405",
    "M7":   "NGC_6475",
    "M11":  "NGC_6705",
    "M18":  "NGC_6613",
    "M21":  "NGC_6531",
    "M23":  "NGC_6494",
    "M25":  "IC_4725",
    "M26":  "NGC_6694",
    "M29":  "NGC_6913",
    "M34":  "NGC_1039",
    "M35":  "NGC_2168",
    "M36":  "NGC_1960",
    "M37":  "NGC_2099",
    "M38":  "NGC_1912",
    "M39":  "NGC_7092",
    "M41":  "NGC_2287",
    "M44":  "NGC_2632",
    "M45":  "Melotte_22",
    "M46":  "NGC_2437",
    "M47":  "NGC_2422",
    "M48":  "NGC_2548",
    "M50":  "NGC_2323",
    "M52":  "NGC_7654",
    "M67":  "NGC_2682",
    "M93":  "NGC_2447",
    "M103": "NGC_581",
}

def normalize_cluster_name(user_input: str) -> str:
    """
    Normalize user input to match the catalog format.
    Handles Messier names, missing underscores in NGC/IC,
    and checks if the result is a valid open cluster.
    """
    name = user_input.strip().upper().replace(" ", "")
    
    # Check if it's a Messier designation
    if name in messier_to_catalog:
        candidate = messier_to_catalog[name]
    else:
        # Fix NGC/IC/etc. names missing underscore
        candidate = name
        for prefix in ["NGC", "IC", "COLLINDER", "MELOTTE"]:
            if candidate.startswith(prefix) and "_" not in candidate:
                head = ''.join([c for c in candidate if not c.isdigit()])
                tail = ''.join([c for c in candidate if c.isdigit()])
                if head and tail:
                    candidate = head + "_" + tail
                break
    
    # Check against catalog
    if candidate in catalog_clusters:
        return candidate
    else:
        return f"'{user_input}' is not an open cluster in this catalog."

# Input from user
user_in = input("Enter cluster name: ")
print("Result:", normalize_cluster_name(user_in))
