"""
python generate_data_slidingwindow.py --dataname Clothing14
python generate_data_slidingwindow.py --dataname Movie18
python generate_data_slidingwindow.py --dataname Office14

python generate_data_slidingwindow.py --dataname Toy14
python generate_data_slidingwindow.py --dataname Game18
python generate_data_slidingwindow.py --dataname Game14

python generate_data_slidingwindow.py --dataname Music14
"""

import argparse
import ast
import copy
from datetime import datetime
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
from transformers import AutoTokenizer


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
        min_len=5,
        max_len=10,
        model="../Llama-3.2-3B/",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.valid_start = valid_start
        self.test_start = test_start
        
        self.min_len = min_len
        self.max_len = max_len

        self.save_data_path = save_data_path
        os.makedirs(save_data_path, exist_ok=True)

        print(f"Start processing {dataname} data...")
        self.reviews = self.read_reviews_json_to_df(interactions_data_path)
        self.asin2title = self.get_asin2title(meta_data_path)  # asin -> title
        # self.asin2title, self.asin2meta = self.get_asin2title_metadata(meta_data_path)
        print("Finish reading data")

        self.reviews = self.filter_asin_wo_title(self.reviews)  # only keep asin with title

        if "Movie" in os.path.basename(dataname):
            # selected_asins = np.random.choice(self.reviews["asin"].unique(), size=10000, replace=False)  # Movie_V2
            selected_asins = np.random.choice(self.reviews["asin"].unique(), size=20000, replace=False)  # Movie_V3
            self.reviews = self.reviews[self.reviews["asin"].isin(selected_asins)]

        self.reviews = self.process_k_core(self.reviews, k=self.min_len)

        self.user_to_cid, self.item_to_cid = self.generate_cid()  # user -> cid, item -> cid

        self.max_user_cid = max(self.user_to_cid.values())
        self.max_item_cid = max(self.item_to_cid.values())
        self.reviews = self.add_cid_column(self.reviews)
        print(f"User Num {self.max_user_cid}, Item num {self.max_item_cid}, Interaction Num {len(self.reviews)}")

        # self.cid2meta = self.generate_meta_items()  # 保存每个item的meta信息

        self.cid2title4LLM, self.cid2title4Rec = self.generate_itemcid2title()  # item cid -> title
        self.user_interacted_items_dict = self.get_interacted_items_dict()  # 存储每个user交互过的item序列

    def get_asin2title(self, file_path):
        df = pd.read_csv(file_path)

        asin2title = {}
        for line in tqdm(df.itertuples(), total=len(df)):
            if type(line.name) == str and len(line.name) > 1:
                name = " ".join(line.name.split())
                asin2title[line.id] = self.tokenizer.decode(self.tokenizer.encode(name, add_special_tokens=False))

        return asin2title

    def read_reviews_json_to_df(self, file_path):
        df = pd.read_csv(file_path)
        selected_columns = ["user_id", "recipe_id", "date", "rating"]
        df = df[selected_columns]
        df = df.dropna(subset=selected_columns)
        return df

    def process_k_core(self, df, k):
        while True:
            user_counts = df["user_id"].value_counts()
            item_counts = df["recipe_id"].value_counts()

            less_than_k_user = user_counts[user_counts < k].index
            less_than_k_item = item_counts[item_counts < k].index

            if len(less_than_k_user) == 0 and len(less_than_k_item) == 0:
                break

            df = df[~df["user_id"].isin(less_than_k_user)]
            df = df[~df["recipe_id"].isin(less_than_k_item)]

        return df

    def filter_asin_wo_title(self, df):
        return df[df["recipe_id"].isin(self.asin2title.keys())]

    def add_cid_column(self, df):
        df["user_ID"] = df["user_id"].map(self.user_to_cid)
        df["item_ID"] = df["recipe_id"].map(self.item_to_cid)
        return df

    def generate_cid(self):
        users_list = self.reviews["user_id"].unique().tolist()
        items_list = self.reviews["recipe_id"].unique().tolist()

        user2id = {user: idx + 1 for idx, user in enumerate(users_list)}
        item2id = {item: idx + 1 for idx, item in enumerate(items_list)}

        return user2id, item2id

    def generate_itemcid2title(self):
        cid2title4LLM = {}  # cid -> title 不存在多个cid对应同一个title的情况
        cid2title4Rec = {}  # cid -> title 存在多个cid对应同一个title的情况

        for asin, cid in self.item_to_cid.items():
            title = self.asin2title[asin]
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
            user, item = row.user_ID, row.item_ID
            if user not in users:
                users[user] = {"items": [], "timestamps": []}
            users[user]["items"].append(item)
            users[user]["timestamps"].append(datetime.strptime(row.date, "%Y-%m-%d").timestamp() * 1000)

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

            for i in range(min(self.max_len, len(items) - 1), len(items)):
                st = max(i - self.max_len, 0)
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
    parse.add_argument("--valid-start", type=float, default=0.8)
    parse.add_argument("--test-start", type=float, default=0.9)
    parse.add_argument("-m", "--min-len", type=int, default=5)
    parse.add_argument("-M", "--max-len", type=int, default=10)
    args = parse.parse_args()

    data_path = "Food.com"
    save_data_path = "Food.com" + (f"_{args.min_len}_{args.max_len}" if args.min_len != 5 or args.max_len != 10 else "")

    meta_data_path = os.path.join(data_path, "RAW_recipes.csv")
    interactions_data_path = os.path.join(data_path, "RAW_interactions.csv")
    

    data = Data(
        dataname="Food.com",
        meta_data_path=meta_data_path,
        interactions_data_path=interactions_data_path,
        save_data_path=save_data_path,
        valid_start=args.valid_start,
        test_start=args.test_start,
        min_len=args.min_len,
        max_len=args.max_len,
    )
    data.generate_interactions_data()
    print("Done!")
