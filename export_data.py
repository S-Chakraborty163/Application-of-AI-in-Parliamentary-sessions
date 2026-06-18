import duckdb
conn = duckdb.connect('parliament_v2.duckdb')
conn.execute("COPY (SELECT * FROM synthetic_fqg_dataset) TO 'custom_fqg_training_data.jsonl' (FORMAT JSON);")
print("Export Successful!")
