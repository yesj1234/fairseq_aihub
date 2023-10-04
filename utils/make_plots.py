import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

# plt.rc('font', family='NanumBarunGothic')  # for windows
plt.rc('font', family='AppleGothic')  # for MAC


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsons", help="root folder path containing jsons")
    parser.add_argument(
        "--save_dir", help="root folder path to save generated graphs. Pass an absolute path", default=os.path.curdir)
    parser.add_argument(
        "--from_csv", help="whether to generate plots from existing dataframe.")
    args = parser.parse_args()
    # 전체 데이터 받아오기
    if args.jsons:
        jsons = []
        for root, dir, files in os.walk(args.jsons):
            if files:
                for file in files:
                    _, ext = os.path.splitext(file)
                    if ext == ".json":
                        with open(os.path.join(root, file), "r+", encoding="utf-8") as f:
                            json_file = json.load(f)
                            jsons.append(json_file)

        df = pd.DataFrame(jsons)
        df.to_csv(f"{args.jsons}/file.csv", mode="w", encoding="utf-8")
    if args.from_csv:
        # dataframe from existing csv file
        df = pd.read_csv(args.from_csv)

    # 성별 분포
    def hex_to_RGB(hex_str):
        """ #FFFFFF -> [255,255,255]"""
        # Pass 16 to the integer function for change of base
        return [int(hex_str[i:i+2], 16) for i in range(1, 6, 2)]

    def get_color_gradient(c1, c2, n):
        """
        Given two hex colors, returns a color gradient
        with n colors.
        """
        assert n > 1
        c1_rgb = np.array(hex_to_RGB(c1))/255
        c2_rgb = np.array(hex_to_RGB(c2))/255
        mix_pcts = [x/(n-1) for x in range(n)]
        rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
        return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]

    gender_plot_y = ["여성", "남성"]
    gender_plot_x = [len(df[df["speaker_gender"] == "여성"]),
                     len(df[df["speaker_gender"] == "남성"])]

    def thousands(x, pos):
        return f'{x:,.0f}'

    def gender_plot(x, y):
        fig, ax = plt.subplots(figsize=(11, 3))
        y_pos = np.arange(len(x))
        colors = get_color_gradient("#8A5AC2", "#3575D5", len(x))
        hbars = ax.barh(y_pos, y,
                        color=colors,
                        height=0.6,
                        align="center")
        ax.set_yticks(y_pos, x)
        ax.set_xlabel("명", fontsize=12)
        ax.set_ylabel("성별", fontsize=12)
        ax.set_title(label="성별 분포", fontsize=15)
        ax.invert_yaxis()
        ax.bar_label(hbars, fmt="{:,.0f}", fontsize=10)
        x_labels = ax.get_xticklabels()
        y_labels = ax.get_yticklabels()
        plt.setp(x_labels, fontsize=10)
        plt.setp(y_labels, fontsize=10)
        plt.rcParams.update({"figure.autolayout": True})
        ax.xaxis.set_major_formatter(thousands)
        fig.savefig(f"{args.save_dir}/성별분포.png",
                    transparent=False, dpi=80, bbox_inches="tight")

    gender_plot(gender_plot_y, gender_plot_x)

    # 플랫폼 분포
    platforms = df["source"].value_counts()
    platform_plot_x = platforms.keys()
    platform_plot_y = platforms.values

    def platform_plot(x, y):
        fig, ax = plt.subplots(figsize=(11, 3))
        y_pos = np.arange(len(x))
        colors = get_color_gradient("#8A5AC2", "#3575D5", len(x))
        hbars = ax.barh(y_pos, y,
                        height=0.6,
                        align="center",
                        color=colors)
        ax.set_yticks(y_pos, x)
        ax.set_xlabel("건", fontsize=12)
        ax.set_ylabel("플랫폼 명", fontsize=12)
        ax.set_title(label="플랫폼 분포", fontsize=15)
        ax.invert_yaxis()  # 순서 변경
        ax.bar_label(hbars, fmt="{:,.0f}", fontsize=10)
        x_labels = ax.get_xticklabels()  # ax.set_xticklabels()
        y_labels = ax.get_yticklabels()  # ax.set_yticklabels()
        plt.setp(x_labels, fontsize=10)  # 혹은 setp 로 여러 설정 한번에 하기
        plt.setp(y_labels, fontsize=10)  # 혹은 setp로 여러 설정 한번에 하기
        plt.rcParams.update({"figure.autolayout": True})
        ax.xaxis.set_major_formatter(thousands)
        fig.savefig(f"{args.save_dir}/플랫폼분포.png",
                    transparent=False, dpi=80, bbox_inches="tight")  # 저장

    platform_plot(platform_plot_x, platform_plot_y)

    # 화자수 분포
    speaker_data = df["li_total_speaker_num"].value_counts()
    speaker_plot_x = speaker_data.keys()
    speaker_plot_y = speaker_data.values

    def speaker_plot(x, y):
        fig, ax = plt.subplots(figsize=(11, 3))
        y_pos = np.arange(len(x))
        colors = get_color_gradient("#8A5AC2", "#3575D5", len(x))
        hbars = ax.barh(y_pos, y,
                        height=0.6,
                        align="center",
                        color=colors)
        ax.set_yticks(y_pos, x)
        ax.set_xlabel("건", fontsize=12)
        ax.set_ylabel("화자수 (명)", fontsize=12)
        ax.set_title(label="화자 규모 분포", fontsize=15)
        ax.invert_yaxis()  # 순서 변경
        ax.bar_label(hbars, fmt="{:,.0f}", fontsize=10)
        x_labels = ax.get_xticklabels()  # ax.set_xticklabels()
        y_labels = ax.get_yticklabels()  # ax.set_yticklabels()
        plt.setp(x_labels, fontsize=10)  # 혹은 setp 로 여러 설정 한번에 하기
        plt.setp(y_labels, fontsize=10)  # 혹은 setp로 여러 설정 한번에 하기
        plt.rcParams.update({"figure.autolayout": True})
        ax.xaxis.set_major_formatter(thousands)
        fig.savefig(f"{args.save_dir}/화자규모분포.png",
                    transparent=False, dpi=80, bbox_inches="tight")  # 저장

    speaker_plot(speaker_plot_x, speaker_plot_y)

    # 전사 텍스트 어절 수 분포
    def get_word_phrase(origin_lang, tc_text):
        if origin_lang == "한국어" or origin_lang == "영어":
            return len(tc_text.strip().split(" "))
        else:
            return len(tc_text)

    word_phrase_plot_data = dict()
    for origin_lang, tc_text in df[["origin_lang", "tc_text"]].values:
        word_phrase = get_word_phrase(origin_lang, tc_text)
        if word_phrase_plot_data.get(word_phrase):
            word_phrase_plot_data[word_phrase] += 1
        else:
            word_phrase_plot_data[word_phrase] = 1

    word_phrase_plot_data = dict(
        sorted(word_phrase_plot_data.items(), key=lambda x: x[1], reverse=True))
    word_phrase_plot_x = word_phrase_plot_data.keys()
    word_phrase_plot_y = word_phrase_plot_data.values()

    def word_phrase_plot(x, y):
        fig, ax = plt.subplots(figsize=(20, 15))
        y_pos = np.arange(len(x))
        colors = get_color_gradient("#8A5AC2", "#3575D5", len(x))
        hbars = ax.barh(y_pos, y,
                        height=0.6,
                        align="center",
                        color=colors)
        ax.set_yticks(y_pos, x)
        ax.set_xlabel("건", fontsize=12)
        ax.set_ylabel("어절 수", fontsize=12)
        ax.set_title(label="전사텍스트 어절 수 분포", fontsize=15)
        ax.invert_yaxis()  # 순서 변경
        ax.bar_label(hbars, fmt="{:,.0f}", fontsize=10)
        x_labels = ax.get_xticklabels()  # ax.set_xticklabels()
        y_labels = ax.get_yticklabels()  # ax.set_yticklabels()
        plt.setp(x_labels, fontsize=10)  # 혹은 setp 로 여러 설정 한번에 하기
        plt.setp(y_labels, fontsize=10)  # 혹은 setp로 여러 설정 한번에 하기
        plt.rcParams.update({"figure.autolayout": True})
        ax.xaxis.set_major_formatter(thousands)
        fig.savefig(f"{args.save_dir}/전사텍스트어절수분포.png",
                    transparent=False, dpi=80, bbox_inches="tight")  # 저장

    word_phrase_plot(word_phrase_plot_x, word_phrase_plot_y)

    # 도메인 분포
    DOMAIN_DISTRIBUTION_KO = {
        "일상/소통": 0.2,
        "여행": 0.15,
        "게임": 0.15,
        "경제": 0.05,
        "교육": 0.05,
        "스포츠": 0.05,
        "라이브커머스": 0.15,
        "음식/요리": 0.2
    }
    DOMAIN_DISTRIBUTION_EN = {
        "일상/소통": 0.2,
        "여행": 0.2,
        "게임": 0.2,
        "음식/요리": 0.2,
        "운동/건강": 0.2
    }
    DOMAIN_DISTRIBUTION_CH = {
        "일상/소통": 0.2,
        "여행": 0.2,
        "게임": 0.2,
        "라이브커머스": 0.2,
        "패션/뷰티": 0.2
    }
    DOMAIN_DISTRIBUTION_JP = {
        "일상/소통": 0.2,
        "여행": 0.2,
        "게임": 0.2,
        "음식/요리": 0.2,
        "패션/뷰티": 0.2
    }
    TOTAL = 400000  # 40만 문장

    def get_percent(category, current_count):
        domain, origin_lang, _ = category.split("_")
        if origin_lang.strip().upper() == "KO":
            return (current_count / (DOMAIN_DISTRIBUTION_KO[domain] * TOTAL)) * 100
        elif origin_lang.strip().upper() == "EN":
            return (current_count / (DOMAIN_DISTRIBUTION_EN[domain] * TOTAL)) * 100
        elif origin_lang.strip().upper() == "JP":
            return (current_count / (DOMAIN_DISTRIBUTION_JP[domain] * TOTAL)) * 100
        else:
            return (current_count / (DOMAIN_DISTRIBUTION_CH[domain] * TOTAL)) * 100
    domain_data = df["category"].value_counts(
    ).sort_values()  # sort in ascending order
    domain_plot_x = domain_data.keys()
    domain_plot_y = domain_data.values
    percent = []
    for category, count in zip(domain_plot_x, domain_plot_y):
        percent.append(get_percent(category, count))

    def domain_plot(x, y, percent):
        fig, ax = plt.subplots(figsize=(15, 3))  # change the figsize manually
        y_pos = np.arange(len(x))
        # first axes to draw in all 100%
        y1 = [100 for _ in range(len(y))]
        ax.barh(y_pos, y1, height=0.6, align="center", color="silver")
        ax.set_yticks(y_pos, x)
        ax.set_xlabel("구축 비율", fontsize=12)
        ax.set_ylabel("카테고리", fontsize=12)
        ax.set_title(label="카테고리 분포", fontsize=15)
        x_labels = ax.get_xticklabels()  # ax.set_xticklabels()
        y_labels = ax.get_yticklabels()  # ax.set_yticklabels()
        plt.setp(x_labels, fontsize=10)  # 혹은 setp 로 여러 설정 한번에 하기
        plt.setp(y_labels, fontsize=10)  # 혹은 setp로 여러 설정 한번에 하기
        # second axes sharing the xaxis
        ax2 = ax.twinx()
        bar_container = ax2.barh(
            y_pos, percent, height=0.6, align="center", color="yellowgreen")
        ax2.set_yticks([])
        ax2.bar_label(bar_container, label=percent,
                      fmt="{:,.2f}%", fontsize=10, label_type="center")
        plt.axvline(x=100, linestyle="--")
        plt.rcParams.update({"figure.autolayout": True})
        fig.savefig(f"{args.save_dir}/카테고리 분포.png",
                    transparent=False, dpi=80, bbox_inches="tight")  # 저장

    domain_plot(x=domain_plot_x, y=domain_plot_y, percent=percent)
