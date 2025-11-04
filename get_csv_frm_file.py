import csv


# 输入输出文件名
# input_file = "./blockwise_load_dump"
input_file = "./layerwise_load_dump"
output_file = f"{input_file}.csv"

def convert_tsv_to_csv(in_file, out_file):
    with open(in_file, "r", encoding="utf-8") as f_in, \
         open(out_file, "w", newline="", encoding="utf-8") as f_out:

        # 检测分隔符（\t 或 多个空格）
        first_line = f_in.readline()
        f_in.seek(0)
        delimiter = "\t" if "\t" in first_line else None

        reader = csv.reader(f_in, delimiter=delimiter, skipinitialspace=True)
        writer = csv.writer(f_out)
        for row in reader:
            # 清除空字符串和首尾空格
            cleaned = [c.strip() for c in row if c.strip() != ""]
            writer.writerow(cleaned)

    print(f"✅ 转换完成：{out_file}")

import pandas as pd
import re


# 3. 解析带宽列，例如 "41.46 GiB/s"
def parse_bandwidth(text):
    if not isinstance(text, str):
        return None
    m = re.match(r"([\d\.]+)\s*(\w+)?/s", text)
    if not m:
        return None
    value, unit = m.groups()
    value = float(value)
    unit = (unit or "").lower()

    if unit.startswith("gib"):
        return value * (1024**3)
    elif unit.startswith("gb"):
        return value * (1e9)
    elif unit.startswith("mib"):
        return value * (1024**2)
    elif unit.startswith("mb"):
        return value * (1e6)
    else:
        return value  # 没单位就按字面值

# ========== 解析 Duration ==========
def parse_duration(text):
    if not isinstance(text, str):
        return None
    m = re.match(r"([\d\.]+)\s*(\w+)", text)
    if not m:
        return None
    value, unit = m.groups()
    value = float(value)
    unit = unit.lower()
    if unit.startswith("s"):      # 秒
        return value
    elif unit.startswith("ms"):   # 毫秒
        return value * 1e-3
    elif "μ" in unit or "us" in unit:  # 微秒
        return value * 1e-6
    elif unit.startswith("ns"):   # 纳秒
        return value * 1e-9
    return value


def parse_start(text):
    """解析 Start 列如 '2.56567s' → float 秒"""
    if not isinstance(text, str):
        return None
    m = re.match(r"([\d\.]+)\s*s", text)
    return float(m.group(1)) if m else None

if __name__ == "__main__":
    convert_tsv_to_csv(input_file, output_file)
    # 1. 读取 CSV 文件
    df = pd.read_csv(output_file)
    # ========== 解析字段 ==========
    df["Start_s"] = df["Start"].apply(parse_start)
    df["Throughput_BytesPerSec"] = df["Throughput"].apply(parse_bandwidth)
    df["Duration_s"] = df["Duration"].apply(parse_duration)
    df = df[ df["Start_s"] > 2.2]

    # 2. 筛选 CUDA memcpy Device-to-Host
    df_d2h = df[df["Name"].str.contains("Device-to-Host", na=False)]
    # 4. 计算平均带宽

    avg_bw_gib = df_d2h["Throughput_BytesPerSec"].mean() / (1024**3)
    avg_duration_us = df_d2h["Duration_s"].mean() * 1e6

    print(f"✅ CUDA memcpy D2H 平均带宽: {avg_bw_gib:.2f} GiB/s")
    print(f"✅ CUDA memcpy D2H 平均时延: {avg_duration_us:.3f} μs")


    # 2. 筛选 CUDA memcpy Host-to-Device
    df_h2d = df[df["Name"].str.contains("Host-to-Device", na=False)]
    # 4. 计算平均带宽

    avg_bw_gib = df_h2d["Throughput_BytesPerSec"].mean() / (1024 ** 3)
    avg_duration_us = df_h2d["Duration_s"].mean() * 1e6

    print(f"✅ CUDA memcpy H2D 平均带宽: {avg_bw_gib:.2f} GiB/s")
    print(f"✅ CUDA memcpy H2D 平均时延: {avg_duration_us:.3f} μs")

