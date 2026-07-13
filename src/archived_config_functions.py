# def validate_embed_config(embed_config_val, fallback_table_formats):
#     """Parse embed config into a list of EmbeddingsFormat.

#     Items with an explicit format (e.g. "parquet-columns") keep that format.
#     Items without (e.g. "csv") get expanded across all fallback_table_formats.
#     """

#     embed_values = parse_list_values(embed_config_val)
#     results = []
#     for val in embed_values:
#         parts = val.split("-")
#         if len(parts) == 1:
#             results.extend([EmbeddingsFormat(filetype=val, table_format=tf) for tf in fallback_table_formats])
#         elif len(parts) == 2:
#             filetype, table_format = parts
#             results.append(EmbeddingsFormat(filetype=filetype, table_format=table_format))
#         else:
#             raise ValueError(f"Invalid embed config value: {val}. Must be filetype or in the format 'filetype-tableformat'")
#     return results




    # # Validate that dual-format parquet export requires {embeddings_table_format} token
    # parquet_formats = [ef for ef in config['embed'] if ef.filetype == 'parquet']
    # has_columns = any(ef.table_format == 'columns' for ef in parquet_formats)
    # has_serialized = any(ef.table_format == 'serialized' for ef in parquet_formats)
    # if has_columns and has_serialized and '{embeddings_table_format}' not in config['embeddings_output_path_template']:
    #     raise ValueError(
    #         "Exporting both parquet table formats (columns and serialized) requires {embeddings_table_format} token in the embeddings output path template"
    #     )


# @dataclass
# class EmbeddingsFormat:
#     filetype: str = "parquet"
#     table_format: str = "serialized"

#     valid_filetypes: ClassVar[list[str]] = ["parquet", "csv"]
#     valid_table_formats: ClassVar[list[str]] = ["serialized", "columns"]

#     def __init__(self, filetype: str, table_format: str):
#         if filetype not in self.valid_filetypes:
#             raise ValueError(f"Invalid filetype: {filetype}. Valid options are: {self.valid_filetypes}")
#         if table_format not in self.valid_table_formats:
#             raise ValueError(f"Invalid table format: {table_format}. Valid options are: {self.valid_table_formats}")
#         self.filetype = filetype
#         self.table_format = table_format
