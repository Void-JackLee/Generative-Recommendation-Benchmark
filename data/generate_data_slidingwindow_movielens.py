"""
python generate_data_slidingwindow.py --datasize 1m
python generate_data_slidingwindow.py --datasize 10m
"""

import argparse
import ast
import copy
import gzip
import html
import json
import os
import pdb
import random
import re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import csv


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Data:
    def __init__(
        self,
        dataname,
        meta_data_path,
        interactions_data_path,
        save_data_path,
        valid_start=0.8,
        test_start=0.9,
    ):
        self.valid_start = valid_start
        self.test_start = test_start

        self.save_data_path = save_data_path
        os.makedirs(save_data_path, exist_ok=True)

        print(f"Start processing {dataname} data...")
        self.reviews = self.read_reviews_dat_to_df(interactions_data_path)
        self.movieID2title = self.get_movieID2title(meta_data_path)  # movieID -> title
        print("Finish reading data")

        self.reviews = self.filter_movieID_wo_title(self.reviews)  # only keep movieID with title

        self.reviews = self.process_k_core(self.reviews, k=5)

        self.user_to_cid, self.item_to_cid = self.generate_cid()  # user -> cid, item -> cid

        self.max_user_cid = max(self.user_to_cid.values())
        self.max_item_cid = max(self.item_to_cid.values())
        self.reviews = self.add_cid_column(self.reviews)
        print(f"User Num {self.max_user_cid}, Item num {self.max_item_cid}, Interaction Num {len(self.reviews)}")

        self.cid2title4LLM, self.cid2title4Rec = self.generate_itemcid2title()  # item cid -> title
        self.user_interacted_items_dict = self.get_interacted_items_dict()  # 存储每个user交互过的item序列

    def get_movieID2title(self, file_path):
        data = []
        with open(file_path, "r", encoding="latin-1") as file:
            for line in file:
                arr = line.strip().split("::")
                data.append({
                    "MovieID": arr[0],
                    "Title": arr[1],
                    "Genres": arr[2],
                })

        movieID2title = {}
        for meta in tqdm(data):
            if "Title" in meta.keys() and len(meta["Title"]) > 1:
                movieID2title[meta["MovieID"]] = meta["Title"].strip()

        return movieID2title

    def read_reviews_dat_to_df(self, file_path):
        data = {
            "UserID": [],
            "MovieID": [],
            "Rating": [],
            "Timestamp": [],
        }
        with open(file_path, "r", encoding="latin-1") as file:
            for line in file:
                arr = line.strip().split("::")
                data["UserID"].append(arr[0])
                data["MovieID"].append(arr[1])
                data["Rating"].append(float(arr[2]))
                data["Timestamp"].append(int(arr[3]))

        df = pd.DataFrame(data)
        df = df.dropna(subset=["UserID", "MovieID", "Timestamp"])
        return df

    def process_k_core(self, df, k):
        while True:
            user_counts = df["UserID"].value_counts()
            item_counts = df["MovieID"].value_counts()

            less_than_k_user = user_counts[user_counts < k].index
            less_than_k_item = item_counts[item_counts < k].index

            if len(less_than_k_user) == 0 and len(less_than_k_item) == 0:
                break

            df = df[~df["UserID"].isin(less_than_k_user)]
            df = df[~df["MovieID"].isin(less_than_k_item)]

        return df

    def filter_movieID_wo_title(self, df):
        return df[df["MovieID"].isin(self.movieID2title.keys())]

    def add_cid_column(self, df):
        df["user_id"] = df["UserID"].map(self.user_to_cid)
        df["item_id"] = df["MovieID"].map(self.item_to_cid)
        return df

    def generate_cid(self):
        users_list = self.reviews["UserID"].unique().tolist()
        items_list = self.reviews["MovieID"].unique().tolist()

        user2id = {user: idx + 1 for idx, user in enumerate(users_list)}
        item2id = {item: idx + 1 for idx, item in enumerate(items_list)}

        return user2id, item2id

    def generate_itemcid2title(self):
        cid2title4LLM = {}  # cid -> title 不存在多个cid对应同一个title的情况
        cid2title4Rec = {}  # cid -> title 存在多个cid对应同一个title的情况

        for asin, cid in self.item_to_cid.items():
            title = self.movieID2title[asin]
            cid2title4Rec[cid] = title
            if title not in cid2title4LLM.values():
                cid2title4LLM[cid] = title

        item_index_2_title_4_Rec_path = os.path.join(self.save_data_path, "id2name4Rec.json")
        item_index_2_title_path = os.path.join(self.save_data_path, "id2name.json")

        with open(item_index_2_title_path, "w", encoding="utf-8") as f:
            json.dump(cid2title4LLM, f, ensure_ascii=False, indent=4)
        with open(item_index_2_title_4_Rec_path, "w", encoding="utf-8") as f:
            json.dump(cid2title4Rec, f, ensure_ascii=False, indent=4)

        return cid2title4LLM, cid2title4Rec

    def get_interacted_items_dict(self):
        users = dict()

        for row in self.reviews.itertuples():
            user, item = row.user_id, row.item_id
            if user not in users:
                users[user] = {"items": [], "timestamps": []}
            users[user]["items"].append(item)
            users[user]["timestamps"].append(row.Timestamp)

        return users

    def get_interactions(self):
        interactions = []
        users = self.user_interacted_items_dict

        for key in tqdm(users.keys()):
            userid = key

            items = users[key]["items"]
            timestamps = users[key]["timestamps"]
            all = list(zip(items, timestamps))

            res = sorted(all, key=lambda x: int(x[-1]))
            items, timestamps = zip(*res)
            items, timestamps = (list(items), list(timestamps))

            for i in range(min(10, len(items) - 1), len(items)):
                st = max(i - 10, 0)
                interactions.append([userid, items[st : i + 1], int(timestamps[i])])

        interactions = sorted(interactions, key=lambda x: x[-1])

        train_interactions = interactions[: int(len(interactions) * args.valid_start)]
        valid_interactions = interactions[
            int(len(interactions) * args.valid_start) : int(len(interactions) * (args.valid_start + 0.1))
        ]
        test_interactions = interactions[
            int(len(interactions) * (args.test_start)) : int(len(interactions) * (args.test_start + 0.1))
        ]

        return train_interactions, valid_interactions, test_interactions

    def generate_data(self, df_list):
        def save_csv(data, filename, sample_num=-1):
            if sample_num != -1 and len(data) > sample_num:
                data = data.sample(n=sample_num, random_state=42).reset_index(drop=True)
            csv_save_path = os.path.join(
                self.save_data_path, f"{filename}{'_'+str(sample_num) if sample_num!=-1 else ''}.csv"
            )
            data.to_csv(csv_save_path, index=False)

        train_df, valid_df, test_df = df_list

        save_csv(train_df, "train", sample_num=-1)
        save_csv(train_df, "train", sample_num=10000)
        save_csv(train_df, "train", sample_num=100000)
        save_csv(valid_df, "valid", sample_num=-1)
        save_csv(valid_df, "valid", sample_num=5000)
        save_csv(valid_df, "valid", sample_num=500)
        save_csv(test_df, "test", sample_num=-1)
        save_csv(test_df, "test", sample_num=5000)
        save_csv(test_df, "test", sample_num=500)

    def generate_interactions_data(self):
        train_interactions, valid_interactions, test_interactions = self.get_interactions()

        column_names = ["user_id", "item_ids", "timestamp"]
        train_df = pd.DataFrame(train_interactions, columns=column_names)
        valid_df = pd.DataFrame(valid_interactions, columns=column_names)
        test_df = pd.DataFrame(test_interactions, columns=column_names)

        df_list = [train_df, valid_df, test_df]
        self.generate_data(df_list)


if __name__ == "__main__":
    set_seed(42)
    parse = argparse.ArgumentParser()
    parse.add_argument("--datasize", type=str, default="10m")
    parse.add_argument("--valid_start", type=float, default=0.8)
    parse.add_argument("--test_start", type=float, default=0.9)
    args = parse.parse_args()

    data_path = os.path.join(".", f"ml-{args.datasize}")
    save_data_path = os.path.join(".", f"ml-{args.datasize}")
    
    meta_data_path = os.path.join(data_path, "movies.dat")
    interactions_data_path = os.path.join(data_path, "ratings.dat")

    data = Data(
        dataname=f"ml-{args.datasize}",
        meta_data_path=meta_data_path,
        interactions_data_path=interactions_data_path,
        save_data_path=save_data_path,
        valid_start=args.valid_start,
        test_start=args.test_start,
    )
    data.generate_interactions_data()
    print("Done!")
