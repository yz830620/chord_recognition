import streamlit as st
import pandas as pd
import numpy as np
import time
"""
# 小兔觀察指南
Here's our 兔兔情緒起伏 and 成長軌跡去了哪裡:
"""

chart_data = pd.DataFrame(np.random.randn(20, 4), columns=["世紀帝國分數", "肚餓程度", "心情", "抱抱指數"])
st.line_chart(chart_data)

map_data = pd.DataFrame(np.random.randn(200, 2) / [10, 12] + [23.2, 120.3], columns=["lat", "lon"])
map_data1 = pd.DataFrame(np.random.randn(100, 2) / [10, 10] + [25, 121.5], columns=["lat", "lon"])
map_data_all = pd.concat([map_data, map_data1], ignore_index=True)

st.map(map_data_all)

if st.checkbox("Show dataframe"):
    chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])

    st.line_chart(chart_data)
df = pd.DataFrame(
    np.array([[i, j] for i, j in zip(["拍打頭部", "餵食", "抱抱"], ["蘋果", "芭樂", "橘子"])]), columns=["first column", "second q"]
)
option = st.sidebar.selectbox("小兔生氣應該怎麼處理膩?", df["first column"])
option1 = st.sidebar.selectbox("小兔打世紀帝國喜歡吃什麼水果?", df["second q"])

"You selected: ", option
"You selected: ", option1

x = st.sidebar.slider("抱多久好呢", 0, 100)
st.write("You selected", x)

left_column, right_column = st.beta_columns(2)
pressed = left_column.button('喜歡請給讚?')
if pressed:
    right_column.write("我也是100萬喜歡歐!")

expander = st.beta_expander("猜猜點開來有什麼")
expander.write("點開來有驚喜")

# 'Starting a long computation...'

# # Add a placeholder
# latest_iteration = st.empty()
# bar = st.progress(0)

# for i in range(100):
#   # Update the progress bar with each iteration.
#   latest_iteration.text(f'Iteration {i+1}')
#   bar.progress(i + 1)
#   time.sleep(0.1)

# '...and now we\'re done!'

progress_bar = st.progress(0)
status_text = st.empty()
chart = st.line_chart(np.random.randn(10, 2))

for i in range(100):
    # Update progress bar.
    progress_bar.progress(i + 1)

    new_rows = np.random.randn(10, 2)

    # Update status text.
    status_text.text(
        '充電中，此刻的電流數據: %s' % new_rows[-1, 1])

    # Append data to the chart.
    chart.add_rows(new_rows)

    # Pretend we're doing some computation that takes time.
    time.sleep(0.1)

status_text.text('Done!')
st.balloons()