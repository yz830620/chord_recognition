import os
import send2trash 
from time import time, sleep
from datetime import datetime

import requests
from bs4 import BeautifulSoup
import re
import json
import pandas as pd
import numpy as np
import streamlit as st
from pytube import YouTube
from moviepy.editor import VideoFileClip, AudioFileClip
from tensorflow.keras.models import Sequential , load_model
import librosa

import load_pred


USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:65.0) Gecko/20100101 Firefox/65.0"
headers = {"user-agent" : USER_AGENT}
number_dict = {1:'first', 2:'second',3:'third',4:'forth',5:'fifth',6:'sixth'}

def get_yt_link(query):
    url = f'https://www.google.com/search?q=youtube+{query}'
    print(f'getting info from youtube, searching for {query}')
    
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        resp.encoding = 'UTF-8'
        soup = BeautifulSoup(resp.text, "html.parser")
    search_blocks = soup.find(class_='tF2Cxc')
    return search_blocks.find(class_="yuRUbf").a.get('href')

def get_lyric(query):
    data = []
    #query = query.encode(encoding='utf-8')
    url = f'https://www.google.com/search?q=魔鏡歌詞+{query}'
    print(f'getting info from mojim, searching for {query}')
    
    docs = get_docs_by_page(url)
    data.extend(docs)
    print('done')
    data = [i for i in data[0] if i['lyric'] != []]
    if data != []:
        df_song = pd.DataFrame({'lyric':data[-1]['lyric'],'timestemp':data[-1]['timestemp']})
        return df_song
    else:
        print('No lyric exist')
        return False


def get_docs_by_page(url):
    def find_block(blocks):
        song = []
        time_counter = 1
        for song_option in blocks:
            song_name = get_song(song_option)
            #singer = get_singer(song_option)
            link = get_link(song_option)
            print(link)
            lyric = get_content_lyric(link)
            timestemp = get_content_lyric(link, True)
            song.append({'song_name': song_name,
                    'lyric': lyric,
                    'timestemp': timestemp
                    })
            if timestemp != []:
                st.write(time_counter, 'is the result returned')
                break
            st.write(f'The {number_dict[time_counter]} attempt not reach,try next')
            time_counter += 1
            
            
        return song
    
    resp = requests.get(url,verify=False, headers=headers,timeout=2)
    if resp.status_code == 200:
        resp.encoding = 'UTF-8'
        soup = BeautifulSoup(resp.text, "html.parser")
    search_blocks = soup.find_all(class_='tF2Cxc',limit=6)
    options = []
    options.append(find_block(search_blocks))
    return(options)

def get_song(song_option_node):
    return song_option_node.find(class_="LC20lb DKV0Md").span.get_text().replace("※ Mojim.com - 魔鏡歌詞網", "")

def get_link(song_option_node):
    return song_option_node.find(class_="yuRUbf").a.get('href')
    
def get_content_lyric(link, isTime=False):
    resp = requests.get(link,verify=False, headers=headers,timeout=2)
    if resp.status_code == 200:
        resp.encoding = 'UTF-8'
        soup = BeautifulSoup(resp.text, "html.parser")
    content_block = soup.find(class_="fsZx3")
    try:
        p_nodes = content_block.get_text()
        p_nodes = p_nodes.replace('[','\r\n[')
        #print(p_nodes)
        pattern = r'^\[([\w\d:.]*)]([\u4E00-\u9FA50-9,.，。! ()\w：　]*)'
        k = re.findall('^\[([\w\d:.]*)]([\u4E00-\u9FA50-9,.，。! ()\w：　]*)', p_nodes, flags=re.M)
        
        if isTime:
            #process time to be the 
            k = [re.sub('[\u4E00-\u9FA5A-Za-z]+', '00',i[0]) for i in k]
            k = [int(i.split(':')[0])*60 + float(i.split(':')[1]) for i in k]
            return k
        else:
            return [i[1].replace('\u3000','') for i in k]
    except:
        return []

def load_song():
    with open('chord.json','r') as f:
        dicf = json.load(f)
    df_label = pd.DataFrame(dicf['1'],columns = ['start','end','chord'])
    return df_label

def song_tab(df_song, df_label):
    df_song['end_time'] = df_song['timestemp'].shift(-1)
    df_song['chord'] = ''
    start_time = df_label['start_time']
    end_time = df_label['end_time']
    label = df_label['chord']

    for i in range(len(label)):
        df_song.loc[(df_song.timestemp < start_time[i]) & (df_song.end_time > end_time[i]),'chord'] += ' ' + label[i].strip()
    return df_song

def iter_song(df_tab):
    for i in df_tab.iterrows():
        lyric, timestemp, end_time, chord = i[1]
        st.write(chord)
        st.write(lyric)

def download_steam(url):
    st.write('正在初始化下載.....')
    if os.path.exists('music_file.wav'): #若路徑中已有檔案存在將其刪除
        send2trash.send2trash("music_file.wav")
    if os.path.exists("music_file.mp4"):
        send2trash.send2trash("music_file.mp4")
    yt = YouTube(url, on_progress_callback = onProgress)
    video = yt.streams.first()
    video_name = yt.title
    filename = 'music_file'
    video.download(filename=filename)
    clip = VideoFileClip(filename + '.mp4')
    clip.audio.write_audiofile(filename + '.wav', write_logfile=False)
    clip.close()
    st.write('下載音樂完成.....')
    st.write(f'檔案為{filename}.wav')
    return filename

def featurize(filename):
    t5 = time()
    st.write('特徵萃取中.....')
    y, sr = librosa.load(filename+'.wav')
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_all={'chroma_cens':chroma_cens.tolist(),'chroma_cqt':chroma_cqt.tolist(),'chroma_stft':chroma_stft.tolist()}
    with open('chroma.json', 'w', encoding='utf-8') as f:
        json.dump(chroma_all, f)
    t6= time()-t5
    st.write('特徵萃取耗時:', t6)
    return 'chroma.json'


def onProgress(stream, chunk, remains): #進程狀態函式
    total = stream.filesize
    percent = round((total-remains) / total * 100)
    download_bar.progress(percent)

def load_pred(): #處理載入模型 & 預測函式
    #將y轉為辭典索引
    array = pd.read_csv('dict/chord_dict.csv')
    dict_chord = array.to_dict()
    
    #載入模型預測
    t7 = time()
    #model=load_model('modelTDT3.h5') #路徑要改
    model=load_model('model/modelTDT3_2.h5') #路徑要改
    st.write('已載入模型')
    # 讀JSON +選取我們用到的特徵
    with open ('chroma.json', "r") as f:
        data = json.loads(f.read()) 
      # Iterating through the json 
    df1 = pd.DataFrame(data['chroma_cqt']).T
    df2 = pd.DataFrame(data['chroma_cens']).T
    df3 = pd.DataFrame(data['chroma_stft']).T
    df123_features = pd.concat([df1,df2,df3], axis=1, ignore_index=True)
        ## 轉成ndarray 為了丟進模型預測
    X_test=np.array(df123_features)
        #預測結果轉成指定輸出(start_time, end_time, chord的json)
    predict = np.argmax(model.predict(X_test), axis = -1)
    t8 = time()-t7
    st.write('載入+預測耗時',t8)
    
    test_list = [dict_chord['chord'][value] for value in predict] #先建立一個list用來儲存預測出來的和弦
    df_test = pd.DataFrame({'chord': test_list}) #轉成Dataframe要用字典，所以裡面長那樣
        #這步驟比較難懂，總之要找出切換合弦的Frame時間點，把它存成list
    df_test['new'] = df_test['chord'].shift(periods=1)
    mask = df_test['chord'] != df_test['new']
    list_chord_change = list(df_test[mask].index)
      #建立一個最終輸輸出的DataFrame(df_fin)，然後對Start, End, Chord欄位填值
    df_fin=pd.DataFrame() 
    df_fin['start_time']= list_chord_change
    df_fin['end_time'] = df_fin['start_time'].shift(-1)
    df_fin.iloc[-1,1]= len(df_test)
    df_fin['chord']= 'None' #先隨便填，下一步再補值
      
    for i,j in enumerate(list_chord_change):
        df_fin.iloc[i,2] = df_test.iloc[j,0] 
    #把Frame轉成時間:  
    f = 512/22050 
    df_fin.iloc[:,[0,1]]= df_fin.iloc[:,[0,1]]* f
    
    #過濾掉長太短的和弦 (ex1: 1200-->200)
    a=[]
    j=0
    for i in range(1,len(df_fin)):
        if df_fin.iat[i,1]-df_fin.iat[i,0]<=0.2:
            df_fin.iat[j,1]=df_fin.iat[i,1]
            a.append(i)
        else:
            j=i
    df_fin2 = df_fin.drop(a)
    df_fin2=df_fin2.reset_index(drop=True)
    #把相同合弦合併 (ex1:200-->120)
    a=[]
    j=0
    for i in range(1,len(df_fin2)):
        if df_fin2.iat[i,2]==df_fin2.iat[j,2]:
            df_fin2.iat[j,1]=df_fin2.iat[i,1]
            a.append(i)       
        else:
            j=i
    df_fin3 = df_fin2.drop(a)
    df_fin=df_fin3.reset_index(drop=True)
    return df_fin

title = st.text_input('Movie title', '無樂不作')
st.write('The youtube link is', get_yt_link(title))
liric = get_lyric(title)
st.write('The lyric is', liric)
download_bar = st.progress(0)
filename = download_steam(get_yt_link(title))

vid_file = open('music_file.mp4', 'rb').read()
st.video(vid_file)

featurize(filename)
df_fin = load_pred()
song_tab = song_tab(liric,df_fin)

st.write('下方就是歌譜了')
st.write('-=-'*20)
iter_song(song_tab)