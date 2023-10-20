# Import library yang dibutuhkan
import pandas as pd
import numpy as np

# Import data yang dibutuhkan
data = pd.read_csv("arrhythmia.data")

# Eksplorasi data
print(data.head())
print(f"\nRows: {data.shape[0]}, Columns: {data.shape[1]}\n")

# Bisa dilihat nama kolom belum ada, saya akan menambahkan berdasarkan dokumentasi data
# Namun hanya terdapat 279 kolom, Kolom terakhir classes saya ambil kesimpulan karena isinya adalah angka 1-16 yang merupakan kelas yang ingin kita prediksi (menurut dokumentasi)
column_names = ["age", "sex", "height", "weight", "qrs_duration", 'p-r_interval', 'q-t_interval', 't_interval', 'p_interval', 'qrs', 
                't', 'p', 'qrst', 'j', 'heart_rate', 'q_wave', 'r_wave', 's_wave', "r'_wave", "s'_wave", 'number_of_intrinsic_deflections', 
                'existence_of_ragged_r_wave', 'existence_of_diphasic_derivation_of_r_wave', 'existence_of_ragged_p_wave', 'existence_of_diphasic_derivation_of_p_wave', 
                'existence_of_ragged_t_wave', 'existence_of_diphasic_derivation_of_t_wave', 'channel_dii', 'channel_dii', 'channel_dii', 'channel_dii', 'channel_dii', 'channel_dii', 
                'channel_dii', 'channel_dii', 'channel_dii', 'channel_dii', 'channel_dii', 'channel_dii', 'channels_diii', 'channel_diii', 'channel_diii', 'channel_diii', 
                'channel_diii', 'channel_diii', 'channel_diii', 'channel_diii', 'channel_diii', 'channel_diii', 'channel_diii', 'channel_diii', 'channel_avr', 'channel_avr', 
                'channel_avr', 'channel_avr', 'channel_avr', 'channel_avr', 'channel_avr', 'channel_avr', 'channel_avr', 'channel_avr', 'channel_avr', 'channel_avr' ,'channel_avl', 
                'channel_avl', 'channel_avl', 'channel_avl', 'channel_avl', 'channel_avl', 'channel_avl', 'channel_avl', 'channel_avl', 'channel_avl', 'channel_avl', 'channel_avl' ,
                'channel_avf', 'channel_avf', 'channel_avf', 'channel_avf', 'channel_avf', 'channel_avf', 'channel_avf', 'channel_avf', 'channel_avf', 'channel_avf', 'channel_avf', 
                'channel_avf', 'channel_v1', 'channel_v1', 'channel_v1', 'channel_v1', 'channel_v1', 'channel_v1', 'channel_v1', 'channel_v1', 'channel_v1', 'channel_v1', 'channel_v1', 
                'channel_v1' ,'channel_v2', 'channel_v2', 'channel_v2', 'channel_v2', 'channel_v2', 'channel_v2', 'channel_v2', 'channel_v2', 'channel_v2', 'channel_v2', 'channel_v2', 
                'channel_v2' ,'channel_v3','channel_v3', 'channel_v3', 'channel_v3', 'channel_v3', 'channel_v3', 'channel_v3', 'channel_v3', 'channel_v3', 'channel_v3', 'channel_v3', 
                'channel_v3','channel_v4', 'channel_v4', 'channel_v4', 'channel_v4', 'channel_v4', 'channel_v4', 'channel_v4', 'channel_v4', 'channel_v4', 'channel_v4', 'channel_v4' ,
                'channel_v4','channel_v5', 'channel_v5', 'channel_v5', 'channel_v5', 'channel_v5', 'channel_v5', 'channel_v5', 'channel_v5', 'channel_v5', 'channel_v5', 'channel_v5' ,
                'channel_v5','channel_v6', 'channel_v6', 'channel_v6', 'channel_v6', 'channel_v6', 'channel_v6', 'channel_v6', 'channel_v6', 'channel_v6', 'channel_v6', 'channel_v6' ,
                'channel_v6', 'jj_wave', 'q_wave', 'r_wave', 's_wave', "r'_wave", "s'_wave", 'p_wave', 't_wave', 'qrsa', 'qrsta', 'of_channel_dii','of_channel_dii', 'of_channel_dii', 
                'of_channel_dii', 'of_channel_dii', 'of_channel_dii', 'of_channel_dii', 'of_channel_dii', 'of_channel_dii', 'of_channel_dii' , 'of_channel_diii', 'of_channel_diii', 
                'of_channel_diii', 'of_channel_diii', 'of_channel_diii', 'of_channel_diii', 'of_channel_diii', 'of_channel_diii', 'of_channel_diii', 'of_channel_diii','of_channel_avr',
                'of_channel_avr', 'of_channel_avr', 'of_channel_avr', 'of_channel_avr', 'of_channel_avr', 'of_channel_avr', 'of_channel_avr', 'of_channel_avr', 'of_channel_avr' ,
                'of_channel_avl','of_channel_avl', 'of_channel_avl', 'of_channel_avl', 'of_channel_avl', 'of_channel_avl', 'of_channel_avl', 'of_channel_avl', 'of_channel_avl', 'of_channel_avl', 
                'of_channel_avf', 'of_channel_avf', 'of_channel_avf', 'of_channel_avf', 'of_channel_avf', 'of_channel_avf', 'of_channel_avf', 'of_channel_avf', 'of_channel_avf', 'of_channel_avf', 
                'of_channel_v1','of_channel_v1', 'of_channel_v1', 'of_channel_v1', 'of_channel_v1', 'of_channel_v1', 'of_channel_v1', 'of_channel_v1', 'of_channel_v1', 'of_channel_v1' ,
                'of_channel_v2','of_channel_v2', 'of_channel_v2', 'of_channel_v2', 'of_channel_v2', 'of_channel_v2', 'of_channel_v2', 'of_channel_v2', 'of_channel_v2', 'of_channel_v2' ,
                'of_channel_v3','of_channel_v3', 'of_channel_v3', 'of_channel_v3', 'of_channel_v3', 'of_channel_v3', 'of_channel_v3', 'of_channel_v3', 'of_channel_v3', 'of_channel_v3' ,
                'of_channel_v4','of_channel_v4', 'of_channel_v4', 'of_channel_v4', 'of_channel_v4', 'of_channel_v4', 'of_channel_v4', 'of_channel_v4', 'of_channel_v4', 'of_channel_v4' ,
                'of_channel_v5','of_channel_v5', 'of_channel_v5', 'of_channel_v5', 'of_channel_v5', 'of_channel_v5', 'of_channel_v5', 'of_channel_v5', 'of_channel_v5', 'of_channel_v5' ,
                'of_channel_v6', 'of_channel_v6', 'of_channel_v6', 'of_channel_v6', 'of_channel_v6', 'of_channel_v6', 'of_channel_v6', 'of_channel_v6', 'of_channel_v6', 'of_channel_v6', 'classes']
data.columns = column_names

# replacing ? as missing value with NaN, biar bisa dicek null
data = data.replace(['?'], np.NaN)

# Check for null
null = data.isnull().sum()
for i in range(len(data.columns)):
    if null[i] > 0:
        print(f"{data.columns[i]}: {null[i]} ({(null[i]/len(data))*100}%)")
total_cells = np.product(data.shape)
total_missing = null.sum()
print(f"\nTotal missing values: {total_missing} ({(total_missing/total_cells) * 100}%)\n")

# Cleaning
print(data.describe(include=["object"]))
print("Data yang menjadi string karena simbol ? \n")
# Kolom heart_rate seharusnya numeric, saya isi NaN dengan rata2nya
# Karena mau diubah ke int saya ubah NaN jadi nangka yang sangat kecil supaya unik dan bisa direplace dengan rata2nya
data['heart_rate'] = data['heart_rate'].replace([np.NaN], '-999999')
data["heart_rate"] = data["heart_rate"].astype(int)
avg = data['heart_rate'].mean(axis=0)
data['heart_rate'] = data['heart_rate'].replace([-999999], avg)

# Kolom t, p, qrst, j adalah vector angle (derajat), maka saya ubah ke numerik
# lalu saya isi NaN dengan rata2nya
data['t'] = data['t'].replace([np.NaN], '-999999')
data["t"] = data["t"].astype(int)
avg = data['t'].mean(axis=0)
data['t'] = data['t'].replace([-999999], avg)

data['p'] = data['p'].replace([np.NaN], '-999999')
data["p"] = data["p"].astype(int)
avg = data['p'].mean(axis=0)
data['p'] = data['p'].replace([-999999], avg)

data['qrst'] = data['qrst'].replace([np.NaN], '-999999')
data["qrst"] = data["qrst"].astype(int)
avg = data['qrst'].mean(axis=0)
data['qrst'] = data['qrst'].replace([-999999], avg)

data['j'] = data['j'].replace([np.NaN], '-999999')
data["j"] = data["j"].astype(int)
avg = data['j'].mean(axis=0)
data['j'] = data['j'].replace([-999999], avg)


# Bisa dilihat data yang NaN sudah tidak ada
print("Bisa dilihat data yang NaN sudah tidak ada")
null = data.isnull().sum()
for i in range(len(data.columns)):
    if null[i] > 0:
        print(f"{data.columns[i]}: {null[i]} ({(null[i]/len(data))*100}%)")
total_cells = np.product(data.shape)
total_missing = null.sum()
print(f"Total missing values: {total_missing} ({(total_missing/total_cells) * 100}%)\n")

# Save ke CSV :D
# data.to_csv("cleaned_data.csv")
