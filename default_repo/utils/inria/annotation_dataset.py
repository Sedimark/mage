def add_quality_annotations_to_df(data, entity_type, assessed_attrs=None, type=None, context_value=None):
    """
    Add quality annotations to a DataFrame for either instance-level or attribute-level annotations (but not both).

    Parameters:
        data (DataFrame): The flattened NGSI-LD data.
        entity_type (str): The NGSI-LD entity type for quality annotations.
        assessed_attrs (list of str): Attributes to annotate with quality information (if None, annotate entire instance).
        type (str): The default `type` for the DataFrame rows if not already exist.
        context_value (str or list): The value to assign to the `@context` column if it does not exist.

    Returns:
        Pandas DataFrame with additional quality annotation columns.
    """
    annotated_data = data.copy()

    # Ensure the 'type' column exists; if not, create it
    if "type" not in annotated_data.columns:
        annotated_data["type"] = type

    # Ensure the 'id' column exists; if not, create it
    if "id" not in annotated_data.columns:
        annotated_data["id"] = annotated_data.apply(
            lambda row: f"urn:ngsi-ld:{row['type']}:{row.name}", axis=1
        )

    # Handle @context column (optional)
    if context_value is not None:  # Only add @context if context_value is provided
        if "@context" not in annotated_data.columns:
            if isinstance(context_value, list):
                # Apply the list across all rows
                annotated_data["@context"] = [context_value] * len(annotated_data)
            elif isinstance(context_value, str):
                # Apply the string across all rows
                annotated_data["@context"] = context_value

    if assessed_attrs is None:
        # Annotate the entire instance (data point)
        annotated_data["hasQuality.type"] = "Relationship"
        annotated_data["hasQuality.object"] = annotated_data.apply(
            lambda row: f"urn:ngsi-ld:DataQualityAssessment:{entity_type}:{row['id']}", axis=1
        )
    else:
        # Annotate specific attributes
        for attr in assessed_attrs:
            # Identify columns that start with the attribute name
            matching_columns = [col for col in data.columns if col.startswith(attr)]
            if not matching_columns:
                raise ValueError(f"Attribute '{attr}' not found in DataFrame columns.")

            # Add quality annotation for each matching attribute column
            for col in matching_columns:
                base_attr = col.split(".")[0]  # Extract the base attribute name
                quality_type_col = f"{base_attr}.hasQuality.type"
                quality_object_col = f"{base_attr}.hasQuality.object"

                # Add quality columns for the attribute
                annotated_data[quality_type_col] = "Relationship"
                annotated_data[quality_object_col] = annotated_data.apply(
                    lambda row: f"urn:ngsi-ld:DataQualityAssessment:{entity_type}:{row['id']}:{base_attr}",
                    axis=1,
                )

    return annotated_data
