"""
Creates .csv files with taxa occurrences from OBIS and GBIF
"""
import sys
import pandas as pd

from dwc_tools.get_gbif_data import get_gbif_data
from dwc_tools.get_obis_occurrences_by_taxaids import get_obis_occurrences_by_taxaids

seagrass_names = [
    "Halodule wrightii",
    "Halophila engelmannii",
    "Thalassia testudinum",
    "Halophila decipiens",
    "Syringodium filiforme"
]

seagrass_aphia_ids = [143769, 143770, 143751, 143768, 234030]

# bounding box for st andrew bay, florida
area_bbox = [-85.9, 29.7, -85.3, 30.1]

# fetch data
gbif_df = get_gbif_data(seagrass_names, area_bbox=area_bbox)
obis_df = get_obis_occurrences_by_taxaids(seagrass_aphia_ids, area_bbox=area_bbox)

print(f"GBIF data: {gbif_df.columns}")
print(f"OBIS data: {obis_df.columns}")

# combine OBIS and GBIF data
combined_df = pd.concat([obis_df, gbif_df], ignore_index=True)

print("Combined OBIS + GBIF data before cleaning:", combined_df.head())


# === clean & preprocess
combined_df = combined_df.drop_duplicates(
    subset=['decimalLatitude', 'decimalLongitude', 'eventDate']
).dropna(
    subset=['decimalLatitude', 'decimalLongitude', 'eventDate']
).reset_index(drop=True)

print("Combined OBIS + GBIF data after cleaning:", combined_df.head())

combined_df['time'] = pd.to_datetime(combined_df['eventDate'], errors='coerce')
combined_df['year'] = combined_df['time'].dt.year

#combined_df = combined_df[combined_df['year'] >= 2000]
#combined_df = combined_df.reset_index(drop=True)

# === save to csv
fname = "seagrass_occurrences_st_andrew_bay.csv"
combined_df.to_csv(fname, index=False)
print(f"Saved seagrass occurrences to {fname}")