import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import platform

# ===================================
# 구미도시공사 공영주차장 이용현황 EDA
# ===================================

DATA_PATH = Path(r"D:\PythonWorkspace\구미도시공사_공영주차장 이용현황_20240612 (1).csv")

OUT_DIR = DATA_PATH.parent / "eda_parking_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
else:
    plt.rcParams["font.family"] = "DejaVu Sans"

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 180

# CSV 불러오기
try:
    df = pd.read_csv(DATA_PATH, encoding="cp949")
except UnicodeDecodeError:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

# 날짜형 변환
if "주차일시" in df.columns:
    df["주차일시"] = pd.to_datetime(df["주차일시"])
else:
    raise KeyError("'주차일시' 컬럼이 없습니다. 컬럼명을 확인하세요.")

print("=" * 60)
print("[1] csv파일 열지않고 column 확인 df.info")
print("=" * 60)
df.info()

# 데이터 전처리
df["월"] = df["주차일시"].dt.month
df["요일번호"] = df["주차일시"].dt.dayofweek
df["요일"] = df["요일번호"].map({0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"})
df["주말여부"] = np.where(df["요일번호"] >= 5, "주말", "평일")
df["총이용대수"] = df["정기주차 이용대수"] + df["일일주차 이용대수"]
df["회전률지표"] = df["입차대수"] + df["출차대수"]

print("\n" + "=" * 60)
print("[2] 전처리 후 df.info")
print("=" * 60)
df.info()

print("\n" + "=" * 60)
print("[3] 결측치 현황")
print("=" * 60)
print(df.isnull().sum())

print("\n" + "=" * 60)
print("[4] 중복 점검")
print("=" * 60)
print("전체 중복 행 수:", df.duplicated().sum())
print("주차일시-주차장명 기준 중복 수:", df.duplicated(subset=["주차일시", "주차장명"]).sum())

# 요약 통계
annual_by_lot = df.groupby("주차장명")["총이용대수"].sum().sort_values(ascending=False)
monthly = df.groupby("월")["총이용대수"].sum()

type_avg = df.groupby("주차장구분")[["입차대수", "출차대수", "총이용대수"]].mean().round(1)

composition = df.groupby("주차장명")[["정기주차 이용대수", "일일주차 이용대수", "총이용대수"]].sum()
#정기비중과 일일비중은 너무 극명하게 차이나는 부분이 많아서 가독성을 위해 백분률로 표시했습니다
composition["정기비중(백분률)"] = composition["정기주차 이용대수"] / composition["총이용대수"]*100
composition["일일비중(백분률)"] = composition["일일주차 이용대수"] / composition["총이용대수"]*100

per_space = df.groupby("주차장명").agg(
    주차구획수=("주차구획수", "mean"),
    총이용대수=("총이용대수", "sum"),
    관측일수=("주차일시", "nunique"),
    입출차합=("회전률지표", "sum"),
)
per_space["구획당_일평균_총이용"] = per_space["총이용대수"] / per_space["주차구획수"] / per_space["관측일수"]

corr_df = df[["주차구획수", "입차대수", "출차대수", "정기주차 이용대수", "일일주차 이용대수", "총이용대수"]].corr().round(2)

print("\n" + "=" * 60)
print("[5] 연간 총이용대수 상위 10개")
print("=" * 60)
print(annual_by_lot.head(10))

print("\n" + "=" * 60)
print("[6] 월별 총이용대수")
print("=" * 60)
print(monthly)

print("\n" + "=" * 60)
print("[7] 주차장 구분별 평균")
print("=" * 60)
print(type_avg)

print("\n" + "=" * 60)
print("[8] 정기주차 비중 상위 5개")
print("=" * 60)
print(composition.sort_values("정기비중(백분률)", ascending=False).head().round(3))

print("\n" + "=" * 60)
print("[9] 일일주차 비중 상위 5개")
print("=" * 60)
print(composition.sort_values("일일비중(백분률)", ascending=False).head().round(3))

print("\n" + "=" * 60)
print("[10] 구획당 일평균 총이용 상위 10개")
print("=" * 60)
print(per_space.sort_values("구획당_일평균_총이용", ascending=False).head(10).round(2))

print("\n" + "=" * 60)
print("[11] 상관관계")
print("=" * 60)
print(corr_df)

# 그래프용 약식 ID 매핑
lot_map = {
    "공단동주차장": "공단",
    "광평천1주차장": "광1",
    "광평천2주차장": "광2",
    "광평천3주차장": "광3",
    "광평천4주차장": "광4",
    "광평천5주차장": "광5",
    "광평천6주차장": "광6",
    "금오천1주차장": "금1",
    "금오천2주차장": "금2",
    "금오천3주차장": "금3",
    "원평가로1주차장": "원1",
    "원평가로2주차장": "원2",
    "원평가로3주차장": "원3",
    "원평가로4주차장": "원4",
    "원평가로5주차장": "원5",
    "원평가로6주차장": "원6",
    "원평가로7주차장": "원7",
    "원평가로8주차장": "원8",
}
df["LotID"] = df["주차장명"].map(lot_map)

# 주차장 아이디별로 이용대수 정리
annual_plot = df.groupby("LotID")["총이용대수"].sum().sort_values(ascending=True)
comp_plot = df.groupby("LotID")[["정기주차 이용대수", "일일주차 이용대수"]].sum().loc[annual_plot.index]

top5_ids = df.groupby("LotID")["총이용대수"].sum().sort_values(ascending=False).head(5).index.tolist()
heatmap_data = (
    df[df["LotID"].isin(top5_ids)]
    .groupby(["LotID", "월"])["총이용대수"]
    .sum()
    .unstack(1)
    .reindex(index=top5_ids)
    .fillna(0)
)

# 1~12월 표시
heatmap_data = heatmap_data.reindex(columns=range(1, 13), fill_value=0)

weekday_order_en = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekday_order_kr = ["월", "화", "수", "목", "금", "토", "일"]

weekday_plot = df.groupby([df["주차일시"].dt.day_name(), "주차장구분"])["총이용대수"].mean().unstack()
weekday_plot = weekday_plot.reindex(weekday_order_en)
weekday_plot.index = weekday_order_kr

plot_type_avg = type_avg.copy()

#위의 여섯개의 그래프를 하나의 시각자료로 표현해야 가독성이 좋을듯 합니다
fig, axes = plt.subplots(3, 2, figsize=(20, 20))
axes = axes.flatten()

# 주차장별 연간 총이용대수
annual_plot.plot(kind="barh", ax=axes[0])
for i, v in enumerate(annual_plot):
    axes[0].text(v + annual_plot.max() * 0.01, i, f"{int(v):,}", va="center", fontsize=8)

axes[0].set_title("주차장별 연간 총이용대수")   
axes[0].set_xlabel("총이용대수")    
axes[0].set_ylabel("")  

# 월별 총이용대수   
monthly = monthly.reindex(range(1, 13), fill_value=0)   
monthly.plot(marker="o", ax=axes[1])    
#마커위에 이용대수 표시
for x, y in monthly.items():
    axes[1].text(x, y, f"{int(y):,}", ha='center', va='bottom')
axes[1].set_title("월별 총이용대수")
axes[1].set_xlabel("월 (Month)")
axes[1].set_ylabel("총이용대수")
axes[1].set_xticks(range(1, 13))
axes[1].grid(True, alpha=0.3)

# 주차장 구분별 평균 이용 규모
plot_type_avg.plot(kind="bar", ax=axes[2])
for container in axes[2].containers:
    for bar in container:
        height = bar.get_height()
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.0f}",
            ha='center',
            va='bottom',
            fontsize=8
        )

axes[2].set_title("주차장 구분별 평균 이용 규모")
axes[2].set_xlabel("")
axes[2].set_ylabel("평균 건수")
axes[2].legend(loc="upper left", frameon=False)
axes[2].tick_params(axis="x", rotation=0)

# [4] 주차장별 정기/일일 이용 구성->가독성을 위해 숫자값을 위아래로 나눠서 적었습니다.
comp_plot.plot(kind="barh", stacked=True, ax=axes[3])

for i, (idx, row) in enumerate(comp_plot.iterrows()):
    cumulative = 0

    for j, col in enumerate(comp_plot.columns):
        value = row[col]
        x_pos = cumulative + value / 2

        # 🔥 핵심: 위치 분리
        if col == "정기주차 이용대수":
            y_pos = i - 0.15   # 위쪽
        else:
            y_pos = i + 0.15   # 아래쪽

        axes[3].text(
            x_pos,
            y_pos,
            f"{int(value):,}",
            ha='center',
            va='center',
            fontsize=8,
            color='black'
        )

        cumulative += value

axes[3].set_title("주차장별 정기/일일 이용 구성")
axes[3].set_xlabel("이용대수")
axes[3].set_ylabel("")
axes[3].legend(["정기주차", "일일주차"], frameon=False)

# [5] 상위 5개 주차장의 월별 이용 히트맵
im = axes[4].imshow(heatmap_data.values, aspect="auto")
axes[4].set_title("상위 5개 주차장의 월별 이용 히트맵")
axes[4].set_xticks(range(12))
axes[4].set_xticklabels(range(1, 13))
axes[4].set_yticks(range(len(top5_ids)))
axes[4].set_yticklabels(top5_ids)
axes[4].set_xlabel("월(Month)")
fig.colorbar(im, ax=axes[4], shrink=0.8, label="이용대수")

# [6] 요일별 평균 이용 패턴
weekday_plot.plot(marker="o", ax=axes[5])

y_offset = np.nanmax(weekday_plot.to_numpy()) * 0.02

for line in axes[5].lines:
    x_data = line.get_xdata()
    y_data = line.get_ydata()
    color = line.get_color()

    for x, y in zip(x_data, y_data):
        if np.ma.is_masked(y) or pd.isna(y):
            continue

        axes[5].text(
            x,
            y + y_offset,
            f"{int(y):,}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=color
        )


axes[5].set_title("요일별 평균 이용 패턴")
axes[5].set_xlabel("")
axes[5].set_ylabel("평균 총이용대수")
axes[5].grid(True, alpha=0.3)

# 전체 제목
fig.suptitle("구미도시공사 공영주차장 이용현황 EDA 시각화\n2026.03.18 모바일공학과 인턴쉽-양희재", fontsize=18, y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(OUT_DIR / "EDA_graph.png", bbox_inches="tight")
plt.close(fig)

# 분석결과를 하나의 엑셀파일에 여러 sheet로 저장 엑셀라이브러리 썼습니다.
excel_path = OUT_DIR / "EDA_summary.xlsx"

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    
    annual_by_lot.reset_index().to_excel(
        writer, sheet_name="연간이용대수", index=False
    )
    
    monthly.reset_index().to_excel(
        writer, sheet_name="월별이용", index=False
    )
    
    type_avg.reset_index().to_excel(
        writer, sheet_name="주차장구분평균", index=False
    )
    
    composition.round(4).reset_index().to_excel(
        writer, sheet_name="이용구성", index=False
    )
    
    per_space.round(4).reset_index().to_excel(
        writer, sheet_name="구획당이용", index=False
    )
    
    corr_df.reset_index().to_excel(
        writer, sheet_name="상관관계", index=False
    )
print("\n" + "=" * 60)