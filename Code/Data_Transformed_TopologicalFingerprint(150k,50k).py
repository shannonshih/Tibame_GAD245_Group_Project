import duckdb
import pandas as pd
from tqdm import tqdm
import numpy as np # linear algebra
from rdkit import Chem
from rdkit.Chem import AllChem

# 設定檔案路徑
train_path = 'train.parquet'

# 建立 DuckDB 連線
con = duckdb.connect()

# 使用進度條來顯示進度
with tqdm(total=2, desc="Processing Data") as pbar:
    # 查詢第一部分數據
    df2_part1 = con.query(f"""SELECT *
                              FROM parquet_scan('{train_path}')
                              WHERE binds = 0
                              ORDER BY random()
                              LIMIT 150000""").df()
    pbar.update(1)  # 更新進度條

    # 查詢第二部分數據
    df2_part2 = con.query(f"""SELECT *
                              FROM parquet_scan('{train_path}')
                              WHERE binds = 1
                              ORDER BY random()
                              LIMIT 50000""").df()
    pbar.update(1)  # 更新進度條

# 合併兩部分數據
df2 = pd.concat([df2_part1, df2_part2], ignore_index=True)

# 關閉連線
con.close()

# # 確認數據
# print(df1.head())

def smiles_to_topological_torsion_fingerprint(smiles, n_bits=2048):
    """
    將 SMILES 分子結構轉換為拓樸扭轉指紋。
    :param smiles: 分子的 SMILES 表示法
    :param n_bits: 指紋的位元數 (默認為 2048)
    :return: 一個 numpy 數組，表示拓樸扭轉指紋
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=int)
    else:
        generator = AllChem.GetTopologicalTorsionGenerator(fpSize=n_bits)
        return np.array(generator.GetFingerprint(mol), dtype=int)

# 使用進度條轉換 "molecule_smiles" 欄位為拓樸扭轉指紋
tqdm.pandas(desc="Transforming molecule_smiles to Topological Torsion Fingerprint")
df2["molecule_smiles"] = df2["molecule_smiles"].progress_apply(lambda x: smiles_to_topological_torsion_fingerprint(x))



# 對 protein 欄位進行 One-Hot Encoding
protein_one_hot = pd.get_dummies(df2["protein_name"], prefix="protein").astype(int)

# 合併 One-Hot 結果
df2_one_hot = pd.concat([df2, protein_one_hot], axis=1)

# 合併需要的欄位：molecule_smiles, binds, 和經過 One-Hot Encoding 的 protein
df2_one_hot = pd.concat([df2_one_hot[["id", "molecule_smiles", "binds"]], protein_one_hot], axis=1)

# # 可選：移除原始 protein 欄位
# df1_one_hot.drop("protein_name", axis=1, inplace=True)

# # 檢視處理後數據
# print(df2_one_hot)

# 僅保留需要的欄位
columns_to_keep = ["id", "molecule_smiles", "binds"] + protein_one_hot.columns.tolist()
df2_filtered = df2_one_hot[columns_to_keep]

# 儲存處理後的數據
output_filename = "train_transformed__topological(150k,50k).parquet"
df2_filtered.to_parquet(output_filename, index=False)

# output_filename = f"train_transformed_topological(15k,5k).parquet"
# df2.to_parquet(output_filename, index=False)