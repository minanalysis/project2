from wordcloud import WordCloud
import matplotlib.pyplot as plt                   # 시각화 도구
plt.rcParams['font.size'] = 15                    # matplotlib의 글꼴 크기
plt.rcParams['font.family'] = 'NanumGothicCoding' # matplotlib의 글꼴

wordcloud = WordCloud(
    font_path='C:\\Users\\JaeIn\\AppData\\Local\\Microsoft\\Windows\\Fonts\\NanumGothicCoding.ttf',
    background_color='white',
    relative_scaling=0.7
).generate(' '.join(copy['댓글']))

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
