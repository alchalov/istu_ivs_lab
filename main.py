# Импорт библиотек
from time import process_time_ns
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy

import warnings
warnings.filterwarnings("ignore")

import folium # for the map

# Установка параметров графика
sns.set_style('whitegrid')
sns.set_palette('Set2')

# Палитра цветов
my_palette = ["#7A92FF", "#FF7AEF", "#B77AFF", "#A9FF7A", "#FFB27A", "#FF7A7A",
             "#7AFEFF", "#D57AFF", "#FFDF7A", "#D3FF7A"]

# Ипортирование данных
data_2015 = pd.read_csv("./WHinput/2015.csv")
data_2016 = pd.read_csv("./WHinput/2016.csv")
data_2017 = pd.read_csv("./WHinput/2017.csv")
data_2018 = pd.read_csv("./WHinput/2018.csv")
data_2019 = pd.read_csv("./WHinput/2019.csv")

# Чтение данных 2015-2017 годов
data_2015 = data_2015[['Country', 'Happiness Rank', 'Happiness Score', 'Economy (GDP per Capita)', 'Family',
                       'Health (Life Expectancy)', 'Freedom', 'Generosity', 'Trust (Government Corruption)', 
                       'Dystopia Residual']]
data_2016 = data_2016[['Country', 'Happiness Rank', 'Happiness Score', 'Economy (GDP per Capita)', 'Family',
                       'Health (Life Expectancy)', 'Freedom', 'Generosity', 'Trust (Government Corruption)', 
                       'Dystopia Residual']]
data_2017 = data_2017[['Country', 'Happiness.Rank', 'Happiness.Score', 'Economy..GDP.per.Capita.', 'Family',
                       'Health..Life.Expectancy.', 'Freedom', 'Generosity', 'Trust..Government.Corruption.', 
                       'Dystopia.Residual']]               

# Заменяем названия колонок в данных в таблицах 2015-2019 годов
new_names = ['Country', 'Happiness Rank', 'Happiness Score', 'Economy (GDP per Capita)', 'Family',
                       'Health (Life Expectancy)', 'Freedom', 'Generosity', 'Trust (Government Corruption)', 
                       'Dystopia Residual']
data_2015.columns = new_names
data_2016.columns = new_names
data_2017.columns = new_names

# Добавляем колонку с указанием года
data_2015['Year'] = 2015
data_2016['Year'] = 2016
data_2017['Year'] = 2017
data_2018['Year'] = 2018
data_2019['Year'] = 2019

# Объединяем данные 2015-2017 годов
data = pd.concat([data_2015, data_2016, data_2017], axis=0)
old_data = data[['Country', 'Happiness Rank', 'Happiness Score','Economy (GDP per Capita)', 'Family', 
                 'Health (Life Expectancy)', 'Freedom', 'Generosity', 'Trust (Government Corruption)', 'Year']]
old_data.columns = ['Country or region', 'Overall rank', 'Score', 'GDP per capita',
       'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Generosity',
       'Perceptions of corruption', 'Year']

# Объединяем данные 2018-2019 годов
new_data = pd.concat([data_2018, data_2019], axis=0)
columns_titles = ['Country or region', 'Overall rank', 'Score', 'GDP per capita',
       'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Generosity',
       'Perceptions of corruption', 'Year']
new_data = new_data.reindex(columns=columns_titles)

# Сводим таблицу по всем годам в единую таблицу
data = pd.concat([old_data, new_data], axis=0)
print(data.head(5))

# Удалаляем поле данных о коррупции. Оно единственное с пропущенными данными.
data[data['Perceptions of corruption'].isna()]

data.dropna(axis = 0, inplace = True)

# Проверяем есть ли другие пропущенные данные.
plt.figure(figsize = (16,6))
sns.heatmap(data = data.isna(), cmap = 'Blues')

# Размер таблицы
table_size = data.shape
print(f'Размер таблицы {table_size[0]} x  {table_size[1]}')

# Базовая статистика сгруппированная по годам (количество значений, среднее выборки, СКО, минимальное значение, перцентили, максимальное значение)
basic_stats = data.groupby(by='Year')['Score'].describe()
print(basic_stats)

# Сгруппируем значения по годам и средним выборок
grouped_by_year_and_means = data.groupby(by = 'Year')[['Score', 'GDP per capita',
       'Social support', 'Healthy life expectancy',
       'Freedom to make life choices', 'Generosity',
       'Perceptions of corruption']].mean().reset_index()

print(grouped_by_year_and_means)

# Сгруппируем по годам и очкам счастья
grouped_by_year_and_facrtors = pd.melt(frame = grouped_by_year_and_means, id_vars='Year', value_vars=['Score', 'GDP per capita',
       'Social support', 'Healthy life expectancy',
       'Freedom to make life choices', 'Generosity',
       'Perceptions of corruption'], var_name='Factor', value_name='Avg_value')

grouped_by_year_and_facrtors.head()

# Построим график измение средних значений параметров по годам
plt.figure(figsize = (16, 9))

ax = sns.barplot(x = grouped_by_year_and_facrtors[grouped_by_year_and_facrtors['Factor'] != 'Score']['Factor'],
                 y = grouped_by_year_and_facrtors['Avg_value'],
                 palette = my_palette[1:],
                 hue = grouped_by_year_and_facrtors['Year'])

plt.title("Различия в факторах по годам - ", fontsize = 25)
plt.xlabel("Фактор", fontsize = 20)
plt.ylabel("Очки счастья (среднее)", fontsize = 20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.legend(fontsize = 15)

ax.set_xticklabels(['Доход','Семья', 'Здоровье', 'Свобода', 'Щедрость', 'Доверие'])

# 10 самых счастливых стран
country_score_avg = data[data['Year']==2019].groupby(by = ['Country or region'])['Score'].mean().reset_index()
table_best = country_score_avg.sort_values(by = 'Score', ascending = False).head(10)

print(f'10 самых счастливых стран \n {table_best}')

plt.figure(figsize = (16, 9))
sns.barplot(y = table_best['Country or region'], x = table_best['Score'], palette = my_palette)
plt.title("Топ 10 самых счастливых стран в 2019", fontsize = 25)
plt.xlabel("Очки счастья", fontsize = 20)
plt.ylabel("Страна", fontsize = 20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)

# 10 самых несчастных стран
table_worst = country_score_avg.sort_values(by = 'Score', ascending = True).head(10)

print(f'10 самых несчастных стран \n {table_worst}')

plt.figure(figsize = (16, 9))
sns.barplot(y = table_worst['Country or region'], x = table_worst['Score'], palette = my_palette)
plt.title("Топ 10 самых несчастных стран в 2019", fontsize = 25)
plt.xlabel("Очки счастья", fontsize = 20)
plt.ylabel("Страна", fontsize = 20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)

# График нормального распределения счастья в 2019 году
plt.figure(figsize = (16, 9))
sns.distplot(a = country_score_avg['Score'], bins = 20, kde = True, color = "#A9FF7A")
plt.xlabel('Очки счастья', fontsize = 20)
plt.title('График нормального распределения счастья в 2019 году', fontsize = 25)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.xlim((1.5, 8.9))

# Графики распределения остальных параметров в 2019 году
country_factors_avg = data[data['Year'] == 2019].groupby(by = ['Country or region'])[['GDP per capita',
       'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Generosity',
       'Perceptions of corruption']].mean().reset_index()

plt.figure(figsize = (16, 9))
sns.kdeplot(data = country_factors_avg['GDP per capita'], color = "#B77AFF", shade = True)
sns.kdeplot(data = country_factors_avg['Social support'], color = "#FD7AFF", shade = True)
sns.kdeplot(data = country_factors_avg['Healthy life expectancy'], color = "#FFB27A", shade = True)
sns.kdeplot(data = country_factors_avg['Freedom to make life choices'], color = "#A9FF7A", shade = True)
sns.kdeplot(data = country_factors_avg['Generosity'], color = "#7AFFD4", shade = True)
sns.kdeplot(data = country_factors_avg['Perceptions of corruption'], color = "#FF7A7A", shade = True)
plt.xlabel('Очки фактора', fontsize = 20)
plt.title('Графики распределения остальных параметров в 2019 году', fontsize = 25)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.xlim((-0.5, 2.3))
plt.legend(fontsize = 15)

# Вычислим, что влияет на счастье?
c1 = scipy.stats.pearsonr(data['Score'], data['GDP per capita'])
c2 = scipy.stats.pearsonr(data['Score'], data['Social support'])
c3 = scipy.stats.pearsonr(data['Score'], data['Healthy life expectancy'])
c4 = scipy.stats.pearsonr(data['Score'], data['Freedom to make life choices'])
c5 = scipy.stats.pearsonr(data['Score'], data['Generosity'])
c6 = scipy.stats.pearsonr(data['Score'], data['Perceptions of corruption'])

print('Очки счастья + Доход: pearson = ', round(c1[0],2), '   pvalue = ', round(c1[1],4))
print('Очки счастья + Семья: pearson = ', round(c2[0],2), '   pvalue = ', round(c2[1],4))
print('Очки счастья + Здоровье: pearson = ', round(c3[0],2), '   pvalue = ', round(c3[1],4))
print('Очки счастья + Свобода: pearson = ', round(c4[0],2), '   pvalue = ', round(c4[1],4))
print('Очки счастья + Щедрость: pearson = ', round(c5[0],2), '   pvalue = ', round(c5[1],4))
print('Очки счастья + Доверие: pearson = ', round(c6[0],2), '   pvalue = ', round(c6[1],4))

# Создадим матрицу корреляции различных параметров друг на друга
corr = data.corr()

# Сгенерируем маску верхнего треугольника
mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Построим график
f, ax = plt.subplots(figsize=(16, 9))

# Построим карту цветов
cmap = sns.diverging_palette(0, 25, as_cmap=True, s = 90, l = 45, n = 5)

# Совмещаем маску, график и карту цветов
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.title('Что влияет на наше счастье?', fontsize = 25)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)

plt.show()
