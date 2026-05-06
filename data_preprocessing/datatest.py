import pyarrow.parquet as pq

path  = r"testdataset/action/ep.parquet"


table = pq.read_table(path)


# p = Path(path).resolve()
# print(p)

df = table.to_pandas()
# See all available columns
print(df.columns.tolist())

# See structure + dtypes
print(df.dtypes)

# Preview data
print(df['episode_index'].max())
