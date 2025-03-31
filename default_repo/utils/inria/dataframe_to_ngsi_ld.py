import json

# --> Data Mapper
def data_conversion(df, entity_type=None):
    """
    Convert a DataFrame into NGSI-LD format.

    Parameters:
        df (DataFrame): The input DataFrame (from CSV, XLS/XLSX or flattened NGSI-LD JSON).
        entity_type (str): The default entity type to use for CSV data.

    Returns:
        A list of NGSI-LD entities in JSON format.
    """
    ngsi_ld_entities = []

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        entity = {
            "id": row.get("id", f"urn:ngsi-ld:{entity_type}:{index}"),  # Use `id` if present, else generate one
            "type": row.get("type", entity_type),  # Use `type` if present, else fallback to the given `entity_type`
        }

        # Process each column
        for column in df.columns:
            if column in ["id", "type"]:  # Skip 'id' and 'type' since they're already added
                continue

            value = row[column]
            if column == '@context':
                # Handle @context: should be a list of URLs
                if isinstance(value, list):
                    entity[column] = value
                else:
                    try:
                        entity[column] = json.loads(value.replace("'", ""))
                    except json.JSONDecodeError:
                        entity[column] = value
            elif '.' in column:
                # Handle nested attributes (assume NGSI-LD JSON format)
                parts = column.split('.')
                if parts[0] not in entity:
                    entity[parts[0]] = {}
                current_level = entity[parts[0]]
                for part in parts[1:-1]:
                    if part not in current_level:
                        current_level[part] = {}
                    current_level = current_level[part]
                current_level[parts[-1]] = value
            else:
                # Treat as Property for CSV-originating data
                entity[column] = {
                    "type": "Property", ### <--
                    "value": value,
                }

        # Append the constructed entity
        ngsi_ld_entities.append(entity)

    return ngsi_ld_entities