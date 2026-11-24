# Data Dictionary

| Column | Meaning | Role in this project |
|---|---|---|
| Sample_ID | Synthetic sample identifier | Excluded (identifier) |
| Source_Calibration | Source-basis/calibration grouping | Excluded (provenance metadata) |
| State | State context | Input categorical feature |
| District | District context | Input categorical feature |
| Soil_Type | Soil type category | Input categorical feature |
| Crop | Crop category | Input categorical feature |
| pH | Soil pH | Input numeric feature |
| Electrical_Conductivity_dS_m | Electrical conductivity (dS/m) | Input numeric feature |
| Organic_Carbon_g_kg | Organic carbon (g/kg) | Input numeric feature |
| Available_N_kg_ha | Available nitrogen (kg/ha) | Input numeric feature |
| Available_P_kg_ha | Available phosphorus (kg/ha) | Input numeric feature |
| Available_K_kg_ha | Available potassium (kg/ha) | Input numeric feature |
| Sulphur_mg_kg | Sulphur (mg/kg) | Input numeric feature |
| Zinc_mg_kg | Zinc (mg/kg) | Input numeric feature |
| Boron_mg_kg | Boron (mg/kg) | Input numeric feature |
| Iron_mg_kg | Iron (mg/kg) | Input numeric feature |
| Manganese_mg_kg | Manganese (mg/kg) | Input numeric feature |
| Copper_mg_kg | Copper (mg/kg) | Input numeric feature |
| Fertility_Score | Teaching score used to derive the target | Excluded to prevent target leakage |
| Fertility_Class | Low / Medium / High teaching label | Target |
