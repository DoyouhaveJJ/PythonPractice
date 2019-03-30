import pandas as pd
from urllib.request import urlretrieve


def load_data(download=False):
    if download:
        data_path, _ = urlretrieve("http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data", "car.csv")
        print("Downloaded to car.csv")

    # use pandas to view the data structure
    # 自定义列名
    col_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
    data = pd.read_csv("car.csv", names=col_names)#pandas的方法
    return data


def convert2onehot(data):
    return pd.get_dummies(data, prefix=data.columns)#很方便的转化


if __name__ == "__main__":
    data = load_data(download=False)
    new_data = convert2onehot(data)

    print(data.head())
    print("\nNum of data: ", len(data), "\n")  # 1728
    # view data values
    for name in data.keys():
        print(name, pd.unique(data[name]))
    print("\n", new_data.head(2))
    new_data.to_csv("car_onehot.csv", index=False)