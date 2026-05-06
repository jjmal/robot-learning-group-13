import pyarrow.parquet as pq
import pandas as pd

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
print(df.shape)

print(f"{df.loc[0,'action']}")
print(f"{df.loc[0,'observation.state']}")



def save_last_ep_parquet():
    df = pd.read_parquet(r"testdataset/action/full.parquet")

    # Filter to only episode_index == 9
    df_filtered = df[df["episode_index"] == 9]

    # Save the filtered result
    df_filtered.to_parquet(r"testdataset/action/ep.parquet", index=False)

    print(f"Done. Rows kept: {len(df_filtered)}")

