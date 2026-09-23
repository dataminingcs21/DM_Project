import pandas as pd
path = "./dataset"
df = pd.read_csv("./dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

print(df.shape)


print(df.tail)