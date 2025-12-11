"""
Creates .csv files with taxa occurrences from OBIS and GBIF
"""
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

# combine OBIS and GBIF data
combined_df = pd.concat([obis_df, gbif_df], ignore_index=True)
combined_df = combined_df.drop_duplicates().dropna(subset=['lat', 'lon', 'time']).reset_index(drop=True)
print("Combined OBIS + GBIF data:", combined_df.shape)
combined_df.head()

# === clean & preprocess
combined_df = combined_df.drop_duplicates()

combined_df = combined_df.dropna(subset=['lat', 'lon', 'time'])

combined_df['time'] = pd.to_datetime(combined_df['time'], errors='coerce')
combined_df = combined_df.dropna(subset=['time'])
combined_df['year'] = combined_df['time'].dt.year

combined_df = combined_df[combined_df['year'] >= 2000]

combined_df = combined_df.reset_index(drop=True)

# === save to csv
fname = "seagrass_occurrences_st_andrew_bay.csv"
combined_df.to_csv(fname, index=False)
print(f"Saved seagrass occurrences to {fname}")